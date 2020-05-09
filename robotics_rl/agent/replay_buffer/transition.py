from collections import namedtuple


DDPGTransition = namedtuple("Transition", ("state",
                                           "action",
                                           "reward",
                                           "next_state",
                                           "terminal")
                            )

SACTransition = namedtuple("Transition", ("state",
                                          "action",
                                          "logprob",
                                          "reward",
                                          "next_state",
                                          "terminal")
                            )
