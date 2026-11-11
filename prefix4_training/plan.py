from prefix_training.plan import main
from .validate import validate


if __name__ == '__main__':
    main(experiment='a60_prefix4', expected_windows=10308,
         expected_episodes=64, expected_actions=10344, bank_validator=validate)
