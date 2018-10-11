from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector


class DeepPavlov:

    def __init__(self):
        tmp = PatternMatchingSkill(responses=['test'], patterns=['test application'])
        tmp2 = PatternMatchingSkill(responses=['test2'], patterns=['lorem'])
        # we need one function lower since we want to give pattern but not get response back
        # fuck yes we can just ask for response and validate response, if len(response) == 1 its all fine
        self.bot = DefaultAgent([tmp, tmp2], skills_selector=HighestConfidenceSelector())

    def get_intent(self, sentence):
        result = self.bot([sentence])
        if len(result) > 1:
            raise AssertionError('Expected one intent result')
        return result[0]
