from prefix_training.publish import main
from .spec import PREFIX


if __name__ == '__main__':
    main(prefix=PREFIX, expected_counts=(4, 480, 444), expected_total=10308)
