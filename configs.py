# Configurations and globals in here

NUM_CLASSES = 10
IMG_ROWS = 28
IMG_COLS = 28

USING_MODELS = True     # Global flag, if I need to experiment without tensorflow being active

if USING_MODELS:        # Using models, then smaller pop size
    POP_SIZE = 8
    GEN_NUM = 10
else:
    POP_SIZE = 50
    GEN_NUM = 50

# Create a dictionary for each type of label
LABELS = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}





