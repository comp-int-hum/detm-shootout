CUDA_DEVICE = "cuda:0"
BATCH_SIZE = 2048
MIN_WORD_COUNT = 50
MAX_WORD_PROPORTION = 0.5
TOPIC_COUNT = 25
LEARNING_RATE = 0.004
MAX_EPOCHS = 100
MODEL_TYPES = ["xDETM", "xDETMcoeff", "cETM"] #, "xDETMCoeff"] #["cETM", "DBDETM", "xDETM"]

STUDIES = [
    {
        "NAME" : "acl_arc",
        "TIME_FIELD" : "year",
        "CONTENT_FIELD" : "text",
        "WINDOW_SIZE" : 10,
        "DOWN_SAMPLE" : 0,
        "SPLIT_FIELD" : "year",
        "BATCH_SIZE" : 128,
        "LEARNING_RATE" : 0.004,
        "WORD_SIMILARITY_TARGETS" : ["captain", "alien", "ship"],
    },
]
