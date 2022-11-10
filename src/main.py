import os.path

from src.classification.cnn_learning.resnext_lstm import ResNext101LSTM, ResNextLongLSTM
from src.classification.vector_learning import nn_factory
from src.util.torch_dataloader import ImageLoader

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
GLYPH_DIR = os.path.join(DATASET_DIR, "glyphs")

ALL_IMAGE_DIR = os.path.join(IMAGE_DIR, "all")
ALL_RAW_DIR = os.path.join(ALL_IMAGE_DIR, "raw")
ALL_BINARIZED_DIR = os.path.join(ALL_IMAGE_DIR, "binarized")

TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train")
TRAIN_IMAGE_MONO_RAW_DIR = os.path.join(TRAIN_IMAGE_DIR, "mono", "raw")
TRAIN_IMAGE_MONO_BINARY_DIR = os.path.join(TRAIN_IMAGE_DIR, "mono", "binarized")
TRAIN_LABEL_DIR = os.path.join(LABEL_DIR, "train")
TRAIN_LABEL_RAW_DIR = os.path.join(TRAIN_LABEL_DIR, "raw")
TRAIN_LABEL_MONO_RAW_DIR = os.path.join(TRAIN_LABEL_DIR, "mono", "raw")
TRAIN_LABEL_MONO_BINARY_DIR = os.path.join(TRAIN_LABEL_DIR, "mono", "binarized")
EVAL_IMAGE_DIR = os.path.join(IMAGE_DIR, "eval")
EVAL_IMAGE_MONO_RAW_DIR = os.path.join(EVAL_IMAGE_DIR, "mono", "raw")
EVAL_IMAGE_MONO_BINARY_DIR = os.path.join(EVAL_IMAGE_DIR, "mono", "binarized")
EVAL_LABEL_DIR = os.path.join(LABEL_DIR, "eval")
EVAL_LABEL_RAW_DIR = os.path.join(EVAL_LABEL_DIR, "raw")
EVAL_LABEL_MONO_RAW_DIR = os.path.join(EVAL_LABEL_DIR, "mono", "raw")
EVAL_LABEL_MONO_BINARY_DIR = os.path.join(EVAL_LABEL_DIR, "mono", "binarized")
TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, "test")
TEST_LABEL_DIR = os.path.join(LABEL_DIR, "test")

TRAIN_RAW_DIR = os.path.join(TRAIN_IMAGE_DIR, "raw")
EVAL_RAW_DIR = os.path.join(EVAL_IMAGE_DIR, "raw")
TRAIN_BINARIZED_DIR = os.path.join(TRAIN_IMAGE_DIR, "binarized")
EVAL_BINARIZED_DIR = os.path.join(EVAL_IMAGE_DIR, "binarized")

COCO_TRAINING_DIR = os.path.join("HomerCompTraining")
COCO_TESTING_DIR = os.path.join("HomerCompTesting")

RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "raw")
TRAIN_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "raw")
EVAL_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "raw")
BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "binarized")
TRAIN_BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "binarized")
EVAL_BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "binarized")

RAW_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "raw")
BINARIZED_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "binarized")
ARTIFICIAL_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "artificial")

lang_file = os.path.join(DATASET_DIR, "perseus_25000.txt"), os.path.join(DATASET_DIR, "perseus_5000.txt")
meta_data = os.path.join(GLYPH_DIR, "meta.csv")


# TRAIN_RESIZED_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "resized", "raw")
# TRAIN_RESIZED_BINARIZED_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "resized", "binarized")
# TRAIN_OCULAR_TRAIN_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "ocular_training")
# TRAIN_OUTPUT_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "output")
# RAW_OCULAR_GLYPHS_DIR = os.path.join(GLYPH_DIR, "ocular", "raw")
# BINARIZED_OCULAR_GLYPHS_DIR = os.path.join(GLYPH_DIR, "ocular", "binarized")


def train_model():
    # TODO: Friday Night, try 101 lstm with correct transforms
    # nn_factory.train_model(lang_file, meta_data,
    #                        TRAIN_RAW_GLYPHS_DIR, EVAL_RAW_GLYPHS_DIR,
    #                        ResNext101LSTM,
    #                        epochs=200, batch_size=8, num_workers=0, resume=True, start_epoch=55, loader=ImageLoader,
    #                        transforms=[ResNext101LSTM.transform_train, ResNext101LSTM.transform_classify])
    # TODO: Thursday Night, continue after removing softmax
    nn_factory.train_model(lang_file, meta_data,
                           TRAIN_RAW_GLYPHS_DIR, EVAL_RAW_GLYPHS_DIR,
                           ResNextLongLSTM,
                           epochs=200, batch_size=8, num_workers=0, resume=True, start_epoch=14, loader=ImageLoader,
                           transforms=[ResNext101LSTM.transform_train, ResNext101LSTM.transform_classify])


if __name__ == '__main__':
    # data_extraction.generate_yolo_labels(COCO_TRAINING_DIR, TRAIN_LABEL_MONO_BINARY_DIR, mono_class=True)
    # dir_util.split_eval_data(TRAIN_IMAGE_MONO_BINARY_DIR,
    #                          TRAIN_LABEL_MONO_BINARY_DIR,
    #                          EVAL_IMAGE_MONO_BINARY_DIR,
    #                          EVAL_LABEL_MONO_BINARY_DIR)

    # data_extraction.extract_glyphs(COCO_TRAINING_DIR,
    #                                TRAIN_IMAGE_MONO_RAW_DIR,
    #                                TRAIN_RAW_GLYPHS_DIR)
    # data_extraction.extract_glyphs(COCO_TRAINING_DIR,
    #                                EVAL_IMAGE_MONO_RAW_DIR,
    #                                EVAL_RAW_GLYPHS_DIR)

    # binarize.cnn(INPUT_DIR, CONFIDENCE_DIR, threshold=None)

    # templates_vector, template_class = alex_init(ARTIFICIAL_TEMPLATE_GLYPHS_DIR)
    # all_vectors, all_classes = alex_init(BINARIZED_GLYPHS_DIR)
    # alex_knn(templates_vector, template_class, all_vectors, all_classes)

    # print("Loading Model...")
    # model, _ = nn_factory.load_model(ResNetLSTM, input_size=0, load_epoch=105, resume=False)
    #
    # brg_img = cv2.imread(
    #     r"C:\Users\Carson Brown\git\glyphs\dataset\images\eval\raw\P_Hamb_graec_665.jpg")
    # # brg_img = cv2.imread(
    # #     r"C:\Users\Carson Brown\git\glyphs\dataset\images\eval\raw\PSI_XIV_1377r.jpg")
    # rgb_img = cv2.cvtColor(brg_img, cv2.COLOR_BGR2RGB)
    #
    # bboxes = yolo.sliding_glyph_window(rgb_img)
    #
    # valid_boxes, outlier_boxes = remove_bbox_outliers(bboxes), get_bbox_outliers(bboxes)
    #
    # lines, (over_size_bboxes, duplicate_bboxes) = bbox_connection.link_bboxes(bboxes)
    #
    # nn_factory.classify(model, lines, rgb_img, ResNetLSTM.transform_classify)
    #
    # for line in lines:
    #     for bbox in line:
    #         print(bbox.get_class())
    #         cv2.imshow("", bbox.crop(brg_img))
    #         cv2.waitKey(0)

    # overlay = img.copy()
    #
    # plot_bboxes(img, valid_boxes, color=(0, 0, 0), wait=None)
    #
    # plot_bboxes(overlay, over_size_bboxes, color=(0, 0, 255), wait=None)
    # plot_bboxes(overlay, duplicate_bboxes, color=(0, 165, 255), wait=None)
    #
    # alpha = .5
    # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    #
    # line_centers = []
    # for line in lines:
    #     line_centers.append([bbox.center for bbox in line])
    #
    # plot_lines(img, line_centers)

    train_model()

    # eval_dataset, eval_dataloader = nn_factory.generate_dataloader(lang_file, meta_data,
    #                                                                EVAL_BINARIZED_GLYPHS_DIR, batch_size=4)
    # model, _ = nn_factory.load_model(
    #     LinearToLSTM, name="binarized", load_epoch=9, dataset=eval_dataset, resume=False)
    # markov_chain = markov.init_markov_chain(os.path.join(DATASET_DIR, "perseus.txt"), get_classes_as_glyphs(),
    #                                         cache_path=os.path.join("dataset", "perseus.markov"))
    #
    # avg_precision, avg_recall, avg_fscore = nn_factory.eval_model(model, eval_dataloader, average="weighted", seed=0)
    # print(avg_precision, avg_recall, avg_fscore)
    # avg_precision, avg_recall, avg_1fscore = nn_factory.eval_model(model, eval_dataloader, average="macro", seed=0)
    # print(avg_precision, avg_recall, avg_fscore)
