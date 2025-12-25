prompt = """More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. 

Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you."""

from ease.predictor_set import PredictorSet
from ease.predictor_extractor import PredictorExtractor


def create_ease_features(answer):

	ps = PredictorSet()

	ps.add_row([], [answer], 2)

	pe = PredictorExtractor()

	pe.initialize_dictionaries(ps)

	x, norm_bag_size, stem_bag_size = pe.gen_feats(ps, prompt)

	# print(type(x))
	# print(x.tolist())
	# print(x.shape)
	# print(norm_bag_size)
	# print(stem_bag_size)

	feature = x

	return feature