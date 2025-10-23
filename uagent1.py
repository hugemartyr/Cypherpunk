from uagents import Agent, Context, Model

# Data model (envolope) which you want to send from one agent to another
class Message(Model):
    message : str
    field : int

my_first_agent = Agent(
    name = 'My First Agent',
    port = 5050,
    endpoint = ['http://localhost:5050/submit']
)

second_agent = 'agent1qw44mpcxdqsa6nxhf3smclcwg5phtt8v00qjy4h60c99msxf6q8253zguu3'

@my_first_agent.on_event('startup')
async def startup_handler(ctx : Context):
    ctx.logger.info(f'My name is {ctx.agent.name} and my address  is {ctx.agent.address}')
    await ctx.send(second_agent, Message(message = 'Hi Second Agent, this is the first agent.'))

@my_first_agent.on_event("shutdown")
async def introduce_agent(ctx: Context):
    ctx.logger.info(f"Hello, I'm agent {ctx.agent.name} and I am shutting down")
    

if __name__ == "__main__":
    my_first_agent.run()