import unittest
from src.agents.biological_expert_agent import BiologicalExpertAgent
from src.agents.experimental_expert_agent import ExperimentalExpertAgent
from src.agents.computational_expert_agent import ComputationalExpertAgent
from src.agents.pi_agent import PI
from src.conversation.orchestrator import ConversationOrchestrator

class TestConversation(unittest.TestCase):

    def setUp(self):
        self.biological_agent = BiologicalExpertAgent()
        self.experimental_agent = ExperimentalExpertAgent()
        self.computational_agent = ComputationalExpertAgent()
        self.pi_agent = PI()
        self.orchestrator = ConversationOrchestrator(
            self.biological_agent,
            self.experimental_agent,
            self.computational_agent,
            self.pi_agent
        )

    def test_agent_responses(self):
        question = "What are the effects of climate change on biodiversity?"
        self.biological_agent.ask_question(question)
        response = self.biological_agent.respond()
        self.assertIsNotNone(response)

        self.experimental_agent.ask_question(question)
        response = self.experimental_agent.respond()
        self.assertIsNotNone(response)

        self.computational_agent.ask_question(question)
        response = self.computational_agent.respond()
        self.assertIsNotNone(response)

    def test_conversation_flow(self):
        self.pi_agent.initiate_conversation("Let's discuss a biological problem.")
        conversation_summary = self.orchestrator.manage_conversation()
        self.assertIn("summary", conversation_summary)

if __name__ == '__main__':
    unittest.main()