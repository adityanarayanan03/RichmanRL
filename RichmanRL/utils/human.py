from evaluation import evaluate_policies
from policy import *

evaluate_policies("ttt", RandomBiddingPolicy(None, 201, 0),
                  RandomGamePolicy(None, 9, 0),
                  RandomBiddingPolicy(None, 201, 0),
                  RandomGamePolicy(None, 9, 0), mode="human")