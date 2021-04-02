import numpy as np
from loguru import logger

ALPHAS = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6]


def build_renyi_pairs(ε, δ):
    r = []
    for α in ALPHAS:
        ε_α = max(ε + np.log(δ) / (α - 1), 0)
        r.append((α, ε_α))
    return r


class PrivacyBudget(object):
    def __init__(self, dict):
        """Initialize from dictionary. To convert back from `to_dict()`

        Args:
            dict ([type]): [description]
        """
        if "renyi" in dict:
            self.__class__ = RenyiBudget
            pairs = []
            for d in dict["renyi"]:
                pairs.append((d["alpha"], d["epsilon"]))
            self.renyi_pairs = pairs

            # self = RenyiBudget(pairs)

        elif "epsDel" in dict:
            self.__class__ = EpsDelBudget
            self.epsilon = dict["epsDel"]["epsilon"]
            self.delta = dict["epsDel"]["delta"]
            # self = EpsDelBudget(dict["epsDel"]["epsilon"], dict["epsDel"]["delta"])

    def empty_copy(self):
        pass

    def is_renyi(self):
        pass

    def to_dict(self):
        pass

    def to_list(self):
        pass


class RenyiBudget(PrivacyBudget):
    def __init__(self, *args):
        """Pass either:
        - Nothing: it initilize an empty RDP vector with the standard alphas, or
        - a list of renyi pairs, such as: [(a_1, e_1), (a_2, e_2)], or
        - two floats, such as: eps, del
        """
        if len(args) == 0:
            self.renyi_pairs = []
            for α in ALPHAS:
                self.renyi_pairs.append((α, 0))
        elif len(args) == 1 and isinstance(args[0], list):
            self.renyi_pairs = args[0]
        elif len(args) == 2:
            self.renyi_pairs = build_renyi_pairs(args[0], args[1])
        else:
            raise Exception(f"Invalid input for Renyi budget: {args}")

    def empty_copy(self):
        pairs = []
        for p in self.renyi_pairs:
            pairs.append((p[0], 0))
        return RenyiBudget(pairs)

    def is_renyi(self):
        return True

    def to_dict(self):
        d = []
        for alpha, epsilon in self.renyi_pairs:
            d.append(
                {
                    "alpha": alpha,
                    "epsilon": epsilon,
                }
            )
        return {"renyi": d}

    def to_list(self):
        """Projects on the standard alphas, can be lossy. Missing alphas are replaced with None.

        Returns:
            list: values of epsilon over the standard ALPHA tracking orders
        """
        d = []
        for base_alpha in ALPHAS:
            # Quadratic
            found = False
            for alpha, epsilon in self.renyi_pairs:
                if abs(alpha - base_alpha) <= 1e-12:
                    d.append(epsilon)
                    found = True
                    break
            if not found:
                d.append(None)

        return d


class EpsDelBudget(PrivacyBudget):
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta

    def empty_copy(self):
        return EpsDelBudget(0, 0)

    def is_renyi(self):
        return False

    def to_dict(self):
        return {
            "epsDel": {
                "epsilon": self.epsilon,
                "delta": self.delta,
            }
        }

    def to_renyi(self) -> RenyiBudget:
        r = build_renyi_pairs(self.epsilon, self.delta)
        return RenyiBudget(r)

    def to_list(self):
        return [self.epsilon, self.delta]


if __name__ == "__main__":
    e = EpsDelBudget(1, 0.001)
    r1 = RenyiBudget()
    r2 = RenyiBudget([(2, 3), (4, 6.7), (1e3, 10)])
    r3 = RenyiBudget(1, 0.001)
    for b in [e, r1, r2, r3]:
        logger.debug(b)
        logger.debug(b.is_renyi())
        logger.debug(b.to_dict())
        logger.debug(b.empty_copy())
        logger.debug(b.empty_copy().to_dict())
        logger.debug(b.to_list())

    a = PrivacyBudget(
        {
            "renyi": [
                {"alpha": 1.5, "epsilon": 0},
                {"alpha": 1.75, "epsilon": 0.7896596280238182},
                {"alpha": 2, "epsilon": 3.092244721017863},
                {"alpha": 2.5, "epsilon": 5.394829814011909},
                {"alpha": 3, "epsilon": 6.546122360508932},
                {"alpha": 4, "epsilon": 7.697414907005955},
                {"alpha": 5, "epsilon": 8.273061180254466},
                {"alpha": 6, "epsilon": 8.618448944203573},
                {"alpha": 8, "epsilon": 9.013177817288266},
                {"alpha": 16, "epsilon": 9.539482981401191},
                {"alpha": 32, "epsilon": 9.777169184548963},
                {"alpha": 64, "epsilon": 9.890353090809807},
                {"alpha": 1000000.0, "epsilon": 9.999993092237814},
            ]
        }
    )

    logger.debug(a)
    logger.debug(a.to_dict())
    logger.debug(a.is_renyi())

    a = PrivacyBudget(
        {
            "epsDel": {"epsilon": 5, "delta": 1e-4},
        }
    )

    logger.debug(a)
    logger.debug(a.to_dict())
    logger.debug(a.is_renyi())