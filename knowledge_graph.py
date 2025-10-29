import logging
from hyperon import MeTTa, E, S, ValueAtom

# Set up logging
logger = logging.getLogger(__name__)

# # This is the threshold our agent will check against
# THRESHOLD_TO_DEPLOY_NEW_AGENT = 5

def print_all_atoms(metta: MeTTa):
    """Helper function to print all atoms in the MeTTa space."""
    try:
        atoms = metta.space().get_atoms()
        if not atoms:
            print("[Knowledge Graph is empty]")
            return
        for atom in atoms:
            print(f"  {atom}")
    except Exception as e:
        print(f"Error printing atoms: {e}")


class OrchestratorKnowledgeGraph:
    """
    Manages the MeTTa knowledge graph for the Orchestrator Agent.
    Handles all logic for finding models, tracking usage,
    and managing specialist agents.
    """
    
    def __init__(self):
        """Initializes the MeTTa instance and loads the initial knowledge."""
        self.metta = MeTTa()
        self._initialize_atoms()

    def _initialize_atoms(self):
        """
        Populates the MeTTa space with our initial knowledge.
        This defines the relationships between tasks, models, counts, and agents.
        """
        logger.info("Initializing MeTTa knowledge graph...")
        
        # --- 1. Task -> Model Mapping ---
        # Format: (= (model-for-task "task_name" "property") "model_id")
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (model-for-task "image-generation" "default") "segmind/tiny-sd")'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (model-for-task "image-generation" "fast") "segmind/tiny-sd")'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (model-for-task "image-generation" "high-quality") "stabilityai/sdxl-turbo")'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (model-for-task "text-generation" "default") "microsoft/Phi-3-mini-4k-instruct")'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (model-for-task "text-summarization" "default") "sshleifer/distilbart-cnn-12-6")'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (model-for-task "sentiment-analysis" "default") "distilbert/distilbert-base-uncased-finetuned-sst-2-english")'
        ))
        # Fixed typo from your example
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (model-for-task "crypto-news" "default") "crypto-news-model-id")'
        ))

        # --- 2. Model -> Usage Count ---
        # Format: (= (usage-count "model_id") 0)
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (usage-count "segmind/tiny-sd") 0)'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (usage-count "stabilityai/sdxl-turbo") 0)'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (usage-count "microsoft/Phi-3-mini-4k-instruct") 0)'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (usage-count "sshleifer/distilbart-cnn-12-6") 0)'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (usage-count "distilbert/distilbert-base-uncased-finetuned-sst-2-english") 0)'
        ))
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (usage-count "crypto-news-model-id") 0)' # Added this for the crypto agent
        ))
        
        # --- 3. Model -> Specialist Agent ---
        # Format: (= (specialist-agent "model_id") "agent_address")
        
        # Fixed typo from your example ("crypt-" -> "crypto-")
        self.metta.space().add_atom(self.metta.parse_single(
            '(= (specialist-agent "crypto-news-model-id") "agent1qfyxlj9lekd4x7yzuyvka5953gcwsataynakphqaf338lmw2u273xhftzyn")'
        ))
        logger.info("Knowledge graph initialized.")

    def find_model_for_task(self, task: str, tag: str = "default") -> str | None:
        """Finds the best model ID for a given task and tag."""
  
        query = f'!(match &self (= (model-for-task "{task}" "{tag}") $model_id) $model_id)'
        result = self.metta.run(query)
        
        # print(f"[DEBUG] find_model_for_task('{task}', '{tag}') result: {result}")
        if result and result[0]:
            # result is like [['"segmind/tiny-sd"']]
            # We take the first result, first element, and convert from Symbol Atom to string
            return str(result[0][0])
        return None

    def find_specialist_agent(self, model_id: str) -> str | None:
        """Finds the address of a specialist agent for a given model ID."""
        
    
        #if(model_id=="<model>"): convert to <model>
        # remove ""
        model_id = model_id.replace('"', '')

        query = f'!(match &self (= (specialist-agent "{model_id}") $agent_addr) $agent_addr)'
        result = self.metta.run(query)
        # print(f"[DEBUG] find_specialist_agent('{model_id}') result: {result}")
        if result and result[0]:
            return str(result[0][0])
        return None

    def get_usage_count(self, model_id: str) -> int:
        """Gets the current usage count for a model ID."""
        # --- FIX: Use !(match &self ...) to query the space ---
        query = f'!(match &self (= (usage-count "{model_id}") $count) $count)'
        result = self.metta.run(query)
        print(f"[DEBUG] get_usage_count('{model_id}') result: {result}")
        if result and result[0]:
            # The result is a ValueAtom, so we get its value
            return int(result[0][0].get_object().value)
        return 0 # Default to 0 if no atom exists

    def increment_usage_count(self, model_id: str) -> int:
        """
        Increments the usage count for a model ID.
        This is a "transaction" (remove old atom, add new one).
        """
        #remove ""
        model_id = model_id.replace('"', '')
        # This function will now work because get_usage_count is fixed
        current_count = self.get_usage_count(model_id)
        new_count = current_count + 1
        
        # 1. Create atoms for the old and new states
        old_atom = self.metta.parse_single(f'(= (usage-count "{model_id}") {current_count})')
        new_atom = self.metta.parse_single(f'(= (usage-count "{model_id}") {new_count})')
        
        # print(f"[DEBUG] increment_usage_count('{model_id}'): {current_count} -> {new_count}")
        
        # 2. Perform the update
        if self.metta.space().remove_atom(old_atom):
            self.metta.space().add_atom(new_atom)
        else:
            # This handles the case where the count atom didn't exist (count was 0)
            self.metta.space().add_atom(new_atom)
            
        return new_count

    def register_specialist_agent(self, model_id: str, agent_address: str):
        """Adds a new atom to register a deployed specialist agent."""
        model_id = model_id.replace('"', '')
        # print(f"[DEBUG] register_specialist_agent({model_id}, '{agent_address}')")
        atom = self.metta.parse_single(f'(= (specialist-agent "{model_id}") "{agent_address}")')
        self.metta.space().add_atom(atom)
        logger.info(f"Registered new specialist for {model_id} at {agent_address}")

    def add_new_task_model(self, task: str, model_id: str, tag: str = "default"):
        """
        Adds a new, unknown task and its model to the knowledge graph.
        This is for the "No i don't have knowledge" case.
        """
        logger.info(f"Adding new task '{task}' with model '{model_id}'")
        # 1. Add the task-to-model mapping
        model_atom = self.metta.parse_single(f'(= (model-for-task "{task}" "{tag}") "{model_id}")')
        self.metta.space().add_atom(model_atom)
        
        # 2. Initialize its usage count to 1 (since it's being used right now)
        count_atom = self.metta.parse_single(f'(= (usage-count "{model_id}") 1)')
        self.metta.space().add_atom(count_atom)
        
    def get_tasks_whom_we_have_knowledge_of(self) -> list[str]:
        """Returns a list of all tasks we have knowledge of."""
    
        query = '!(match &self (= (model-for-task $task_name "default") $model_id) $task_name)'
        result = self.metta.run(query)
        tasks = [str(row[0]) for row in result if row]
        # print(f"[DEBUG] get_tasks_whom_we_have_knowledge_of() result: {tasks}")
        return tasks  
    
    def get_all_specialist_agents(self):
        """Returns a list of all registered specialist agents."""
    
        # query = '!(match &self (= (specialist-agent $model_id) $agent_addr) $model_id $agent_addr)'
        # result = self.metta.run(query)
        # print(f"[DEBUG] get_all_specialist_agents() raw result: {result}")
        # agents = [(str(row[0]), str(row[1])) for row in result if row]
        # # print(f"[DEBUG] get_all_specialist_agents() result: {agents}")
        # return agents
        return []


# --- Main block for Debugging and Testing ---
if __name__ == "__main__":
    import logging
    # Configure logging for the main test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("KG_Test") # This logger is for the __main__ block

    # 1. Initialize the graph
    kg = OrchestratorKnowledgeGraph()
    print("--- 1. KNOWLEDGE GRAPH INITIALIZED ---")
    print_all_atoms(kg.metta)

    # 2. Test: Find model for a known task
    print("\n--- 2. TEST: Find model for 'text-generation' ---")
    model_id = kg.find_model_for_task("text-generation")
    print(f"Found model: {model_id}")

    # 3. Test: Find a specialist agent that *doesn't* exist
    print("\n--- 3. TEST: Find specialist for 'text-generation' model ---")
    agent = kg.find_specialist_agent("crypto-news-model-id")
    print(f"Found agent: {agent}")

    # 4. Test: Find a specialist agent that *does* exist
    print("\n--- 4. TEST: Find specialist for 'crypto-news' model ---")
    model_id_crypto = kg.find_model_for_task("crypto-news")
    print(f"Found model: {model_id_crypto}")
    agent = kg.find_specialist_agent(model_id_crypto)
    print(f"Found agent: {agent}")

    # 5. Test: Increment usage count
    print("\n--- 5. TEST: Increment usage for 'text-generation' model ---")
    count1 = kg.get_usage_count(model_id)
    print(f"Count before: {count1}")
    count2 = kg.increment_usage_count(model_id)
    print(f"Count after: {count2}")
    # Verify by re-reading
    count3 = kg.get_usage_count(model_id)
    print(f"Count from DB: {count3}")

    # 6. Test: Register a new agent
    print("\n--- 6. TEST: Register new specialist for 'text-generation' ---")
    new_agent_addr = "agent1q...newly-deployed-text-gen..."
    kg.register_specialist_agent(model_id, new_agent_addr)
    agent = kg.find_specialist_agent(model_id)
    print(f"Found new agent: {agent}")

    # 7. Test: Add a completely new task
    print("\n--- 7. TEST: Add new task 'sound-generation' (from HF Hub) ---")
    kg.add_new_task_model("sound-generation", "facebook/musicgen-small")
    model = kg.find_model_for_task("sound-generation")
    print(f"Found model: {model}")
    count = kg.get_usage_count("facebook/musicgen-small") # Use the model_id directly
    print(f"Initial count: {count}")
    
    print("\n get all tasks we have knowledge of:")
    tasks = kg.get_tasks_whom_we_have_knowledge_of()
    print(f"Tasks we have knowledge of: {tasks}")

    print("\n--- FINAL KNOWLEDGE GRAPH STATE ---")
    print_all_atoms(kg.metta)