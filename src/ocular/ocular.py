import os

OCULAR_JAR_PATH = os.path.join("tools", "ocular-0.3.jar")

LANGUAGE_MODEL_TXT = os.path.join("dataset", "perseus.txt")
LANGUAGE_MODEL_LMSER = os.path.join("dataset", "lm", "perseus.lmser")
LANGUAGE_MODEL_INIT_FONT = os.path.join("dataset", "font", "perseus-init.fontser")
LANGUAGE_MODEL_TRAINED_FONT = os.path.join("dataset", "font", "perseus-trained.fontser")
LANGUAGE_MODEL_TRAINED_GSM = os.path.join("dataset", "gsm", "perseus-trained.gsmser")


def init_lang_model(*, overwrite=False):
    if not os.path.exists(LANGUAGE_MODEL_LMSER) or overwrite:
        print("Generating Ocular Language Model")
        os.system(
            'java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel '
            '-mx15g '
            f'-jar {OCULAR_JAR_PATH} '
            f'-inputTextPath "greek->{LANGUAGE_MODEL_TXT}" '
            f'-outputLmPath {LANGUAGE_MODEL_LMSER} '
        )


def init_font(*, overwrite=False):
    if not os.path.exists(LANGUAGE_MODEL_INIT_FONT) or overwrite:
        init_lang_model()
        print("Generating Ocular Initial Font")
        os.system(
            'java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.InitializeFont '
            '-mx15g '
            f'-jar {OCULAR_JAR_PATH} '
            f'-inputLmPath {LANGUAGE_MODEL_LMSER} '
            f'-outputFontPath {LANGUAGE_MODEL_INIT_FONT}'
        )


def train_font(training_document_dir, output_dir, *, gsm=False, overwrite=False, num_iters=3):
    if not os.path.exists(LANGUAGE_MODEL_TRAINED_FONT) or overwrite:
        print("Training Ocular Font")
        os.system(
            'java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TrainFont '
            '-mx15g '
            f'-jar {OCULAR_JAR_PATH} '
            f'-inputFontPath {LANGUAGE_MODEL_INIT_FONT} '
            f'-inputLmPath {LANGUAGE_MODEL_LMSER} '
            f'-inputDocPath {training_document_dir} '
            f'-outputFontPath {LANGUAGE_MODEL_TRAINED_FONT} '
            f'-outputPath {output_dir} '
            f'-numEMIters {num_iters} '
            f'-continueFromLastCompleteIteration true '
            f'-beamSize 50 '
            f'-markovVerticalOffset true '
            '' +
            (
                '-allowGlyphSubstitution true '
                '-updateGsm true '
                f'-outputGsmPath {LANGUAGE_MODEL_TRAINED_GSM}'
                if gsm else ''
            )
        )


def init_ocular(training_document_dir, output_dir, gsm=False, overwrite=False, num_iters=50):
    init_lang_model(overwrite=overwrite)
    init_font(overwrite=overwrite)
    train_font(training_document_dir, output_dir, gsm=gsm, overwrite=overwrite, num_iters=num_iters)


def transcribe(training_document_dir, output_dir):
    print("Transcribing With Ocular")
    os.system(
        'java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.Transcribe '
        '-mx15g '
        f'-jar {OCULAR_JAR_PATH} '
        f'-inputDocPath {training_document_dir} '
        f'-inputLmPath {LANGUAGE_MODEL_LMSER} '
        f'-inputFontPath {LANGUAGE_MODEL_TRAINED_FONT} '
        f'-outputPath {output_dir}'
    )
