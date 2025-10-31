"""
uAgent Bridge - REST API with Chat Protocol
Allows Node.js/frontend to send messages to any uAgent using chat protocol
"""
from datetime import datetime
from uuid import uuid4
from uagents import Agent, Protocol, Context, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)
import asyncio
import threading
import time
import requests


# REST API Models
class BridgeRequest(Model):
    """Request format from Node.js/frontend client"""
    target_agent: str  # Dynamic agent address from frontend
    query: str
    request_id: str
    seed: str


class BridgeResponse(Model):
    """Response format to Node.js/frontend client"""
    success: bool
    response: str
    request_id: str
    error: str = ""


# Store pending requests (request_id -> response data)
pending_requests = {}


def register_bridge_with_agentverse(agent_info, bearer_token):
    """Register bridge agent with Agentverse API."""
    try:
        # Only proceed with registration if mailbox is True
        if not agent_info.get("mailbox", True):
            print(
                f"Bridge agent '{agent_info.get('name')}' using endpoint configuration, "
                "skipping Agentverse registration"
            )
            return

        # Wait for agent to be ready
        time.sleep(8)

        agent_address = agent_info.get("address")
        port = agent_info.get("port")
        name = agent_info.get("name")
        description = agent_info.get("description", "Bridge agent for forwarding messages to other uAgents")

        if not agent_address or not bearer_token:
            print("Missing agent address or API token, skipping API calls")
            return

        print(f"Connecting bridge agent '{name}' to Agentverse...")

        # Setup headers
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }

        # Step 1: Connect agent to Agentverse
        connect_url = f"http://127.0.0.1:{port}/connect"
        connect_payload = {
            "agent_type": "mailbox",
            "user_token": bearer_token
        }

        try:
            connect_response = requests.post(
                connect_url, json=connect_payload, headers=headers, timeout=10
            )
            print(f"Connect response: {connect_response.status_code} - {connect_response.text}")
            
            if connect_response.status_code in [200, 201]:
                print(f"Successfully connected bridge agent '{name}' to Agentverse")
            else:
                print(
                    f"Failed to connect bridge agent '{name}' to Agentverse: "
                    f"{connect_response.status_code} - {connect_response.text}"
                )
                return
        except Exception as e:
            print(f"Error connecting bridge agent '{name}' to Agentverse: {str(e)}")
            return

        # Step 2: Register agent with Agentverse API
        print(f"Registering bridge agent '{name}' with Agentverse API...")
        register_url = "https://agentverse.ai/v1/agents"
        
        register_payload = {
            "address": agent_address,
            "agent_type": "mailbox"
        }

        try:
            register_response = requests.post(
                register_url, json=register_payload, headers=headers, timeout=10
            )
            print(f"Register response: {register_response.status_code} - {register_response.text}")
            
            if register_response.status_code in [200, 201]:
                print(f"Successfully registered bridge agent '{name}' with Agentverse API")
            else:
                # Agent might already be registered, try to update instead
                print(f"Registration returned {register_response.status_code}, trying update...")
        except Exception as e:
            print(f"Error registering with Agentverse API: {str(e)}")

        # Step 3: Update agent info on agentverse.ai
        print(f"Updating bridge agent '{name}' README on Agentverse...")
        update_url = f"https://agentverse.ai/v1/agents/{agent_address}"

        # Create README content with badges and input model
        readme_content = f"""# {name}
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
<br />
<br />
{description}
<br />
<br />
vdv
**Input Data Model (BridgeRequest)**
```
class BridgeRequest(Model):
    target_agent: str  # Dynamic agent address
    query: str
    request_id: str
    seed: str
```
**Output Data Model (BridgeResponse)**
```
class BridgeResponse(Model):
    success: bool
    response: str
    request_id: str
    error: str
```
"""

        update_payload = {
            "name": name,
            "readme": readme_content,
            "short_description": description,
        }

        try:
            update_response = requests.put(
                update_url, json=update_payload, headers=headers, timeout=10
            )
            print(f"Update response: {update_response.status_code} - {update_response.text}")
            
            if update_response.status_code == 200:
                print(f"Successfully updated bridge agent '{name}' README on Agentverse")
            else:
                print(
                    f"Failed to update bridge agent '{name}' README on Agentverse: "
                    f"{update_response.status_code} - {update_response.text}"
                )
        except Exception as e:
            print(f"Error updating bridge agent '{name}' README on Agentverse: {str(e)}")

        print(f"Bridge agent '{name}' registration complete!")

    except Exception as e:
        print(f"Error registering bridge agent with Agentverse: {str(e)}")


def create_bridge_agent(seed: str, agentverse_token: str, port: int = None, mailbox: bool = True):
    """
    Create a bridge agent instance for a specific user.
    
    Args:
        seed: Unique seed for the agent
        agentverse_token: Bearer token for Agentverse registration
        port: Port number (optional, will auto-find if not provided)
        mailbox: Whether to use mailbox mode (default: True)
        
    Returns:
        dict: Agent info dictionary with name, address, port, etc.
    """
    from socket import socket
    
    # Auto-find port if not provided
    if port is None:
        port = 8000
        with socket() as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
    
    # Create unique agent name
    agent_name = f"bridge-{seed[:8]}"
    
    # Create the bridge agent
    if mailbox:
        bridge_agent = Agent(
            name=agent_name,
            seed=seed,
            port=port,
            mailbox=True
        )
    else:
        bridge_agent = Agent(
            name=agent_name,
            seed=seed,
            port=port,
            endpoint=[f"http://localhost:{port}/submit"]
        )
    
    # Initialize chat protocol
    chat_proto = Protocol(spec=chat_protocol_spec)
    
    # Agent info storage
    agent_info = {
        "name": agent_name,
        "uagent": bridge_agent,
        "port": port,
        "seed": seed,
        "mailbox": mailbox,
        "api_token": agentverse_token
    }
    
    # Define startup handler to show agent address
    @bridge_agent.on_event("startup")
    async def startup(ctx: Context):
        agent_address = ctx.agent.address
        agent_info["address"] = agent_address
        ctx.logger.info(
            f"Bridge agent '{agent_name}' started with address: {agent_address}"
        )
        
        # Start Agentverse registration in background if we have a token
        if agentverse_token and mailbox:
            threading.Thread(
                target=register_bridge_with_agentverse, 
                args=(agent_info, agentverse_token)
            ).start()

    # REST endpoint to receive requests from Node.js/Frontend
    @bridge_agent.on_rest_post("/query", BridgeRequest, BridgeResponse)
    async def handle_query(ctx: Context, req: BridgeRequest) -> BridgeResponse:
        """
        Receives query from frontend and forwards to target uAgent using chat protocol
        Accepts DYNAMIC agent address from frontend
        """
        try:
            # Create chat message
            message_id = uuid4()
            chat_msg = ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=message_id,
                content=[TextContent(type="text", text=req.query)]
            )
            
            # Store request for response tracking
            pending_requests[str(message_id)] = {
                'request_id': req.request_id,
                'response': None,
                'received': False
            }
            
            # Send message to target agent and wait for response
            await ctx.send(req.target_agent, chat_msg)
            
            # Wait for response using interval message
            response_text = None
            max_wait = 60  # 60 seconds max wait
            check_interval = 0.5  # Check every 500ms
            elapsed = 0
            
            # Store request ID for message handler to find
            pending_requests[str(message_id)] = {
                'request_id': req.request_id,
                'response': None,
                'received': False,
                'sender': None
            }
            
            # Wait for response
            while elapsed < max_wait:
                await asyncio.sleep(check_interval)
                elapsed += check_interval
                
                # Check if we received response
                if pending_requests[str(message_id)]['received']:
                    response_text = pending_requests[str(message_id)]['response']
                    sender = pending_requests[str(message_id)]['sender']
                    
                    # Clean up
                    del pending_requests[str(message_id)]
                    
                    return BridgeResponse(
                        success=True,
                        response=response_text,
                        request_id=req.request_id
                    )
            
            # Timeout - no response received
            del pending_requests[str(message_id)]
            
            return BridgeResponse(
                success=False,
                response="",
                request_id=req.request_id,
                error=f"Timeout waiting for response from target agent"
            )
                
        except Exception as e:
            return BridgeResponse(
                success=False,
                response="",
                request_id=req.request_id,
                error=str(e)
            )

    # Handle incoming chat messages (responses from target agents)
    @chat_proto.on_message(ChatMessage)
    async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
        """Handle incoming chat messages (responses)"""
        # Extract text
        message_text = ""
        for item in msg.content:
            if isinstance(item, TextContent):
                message_text += item.text
        
        # Find matching pending request
        for msg_id, data in pending_requests.items():
            if not data['received']:
                # Store response
                data['response'] = message_text
                data['received'] = True
                data['sender'] = sender
                break
        
        # Send acknowledgement
        ack = ChatAcknowledgement(
            acknowledged_msg_id=msg.msg_id,
            timestamp=datetime.utcnow()
        )
        await ctx.send(sender, ack)

    # Handle acknowledgements
    @chat_proto.on_message(ChatAcknowledgement)
    async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
        """Handle message acknowledgements"""
        pass

    # Include chat protocol
    bridge_agent.include(chat_proto, publish_manifest=True)
    
    # Run agent in background thread
    def run_agent():
        bridge_agent.run()
    
    thread = threading.Thread(target=run_agent)
    thread.daemon = True
    thread.start()
    agent_info["thread"] = thread
    
    # Wait for agent to get address
    wait_count = 0
    while "address" not in agent_info and wait_count < 30:
        time.sleep(0.5)
        wait_count += 1
    
    # Additional wait to ensure agent is fully initialized
    if "address" in agent_info:
        time.sleep(2)
    
    return agent_info


# Default bridge agent (for backward compatibility)
def get_default_bridge_agent():
    """Get or create the default bridge agent"""
    if not hasattr(get_default_bridge_agent, 'agent'):
        get_default_bridge_agent.agent = Agent(
            name="chat-bridge",
            seed="chat-bridge-seed",
            port=8000,
            mailbox=True
        )
        
        chat_proto = Protocol(spec=chat_protocol_spec)
        
        @get_default_bridge_agent.agent.on_event("startup")
        async def startup(ctx: Context):
            pass
        
        @get_default_bridge_agent.agent.on_rest_post("/query", BridgeRequest, BridgeResponse)
        async def handle_query(ctx: Context, req: BridgeRequest) -> BridgeResponse:
            try:
                message_id = uuid4()
                chat_msg = ChatMessage(
                    timestamp=datetime.utcnow(),
                    msg_id=message_id,
                    content=[TextContent(type="text", text=req.query)]
                )
                
                pending_requests[str(message_id)] = {
                    'request_id': req.request_id,
                    'response': None,
                    'received': False,
                    'sender': None
                }
                
                await ctx.send(req.target_agent, chat_msg)
                
                max_wait = 60
                check_interval = 0.5
                elapsed = 0
                
                while elapsed < max_wait:
                    await asyncio.sleep(check_interval)
                    elapsed += check_interval
                    
                    if pending_requests[str(message_id)]['received']:
                        response_text = pending_requests[str(message_id)]['response']
                        del pending_requests[str(message_id)]
                        return BridgeResponse(
                            success=True,
                            response=response_text,
                            request_id=req.request_id
                        )
                
                del pending_requests[str(message_id)]
                return BridgeResponse(
                    success=False,
                    response="",
                    request_id=req.request_id,
                    error=f"Timeout waiting for response from target agent"
                )
            except Exception as e:
                return BridgeResponse(
                    success=False,
                    response="",
                    request_id=req.request_id,
                    error=str(e)
                )
        
        @chat_proto.on_message(ChatMessage)
        async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
            message_text = ""
            for item in msg.content:
                if isinstance(item, TextContent):
                    message_text += item.text
            
            for msg_id, data in pending_requests.items():
                if not data['received']:
                    data['response'] = message_text
                    data['received'] = True
                    data['sender'] = sender
                    break
            
            ack = ChatAcknowledgement(
                acknowledged_msg_id=msg.msg_id,
                timestamp=datetime.utcnow()
            )
            await ctx.send(sender, ack)
        
        @chat_proto.on_message(ChatAcknowledgement)
        async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
            pass
        
        get_default_bridge_agent.agent.include(chat_proto, publish_manifest=True)
    
    return get_default_bridge_agent.agent


if __name__ == "__main__":
    # For testing without parameters
    default_agent = get_default_bridge_agent()
    default_agent.run()