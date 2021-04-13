class params:
    FEATURES = ["Open", "High", "Low", "Close", "Volume"]
    FEATURE_COUNT = 29

    BATCH_SIZE = 128
    WINDOW_SIZE = 10
    SEQUENCE_LENGTH = WINDOW_SIZE // 2
    MOMENTUM_WINDOW_SIZE = WINDOW_SIZE // 2

    EPOCHS = 20
    STEPS_PER_EPOCH = 5653

    VALIDATION_RATIO = 0.2
    VALIDATION_SIZE = 1413

    LAYERS = 2
    D_MODEL = 256
    HEADS = 8
    DFF = D_MODEL // 2
    OPT = 'adam'
