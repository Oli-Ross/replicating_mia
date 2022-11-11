import target_models as tm
import os.path
import shutil
import os


class TestTargetModel():
    def test_load_save_model(self):
        targetModel: tm.KaggleModel = tm.KaggleModel(100)
        tm.save_model("test_model", targetModel)
        curDir = (os.path.dirname(__file__))
        modelFile: str = os.path.join(curDir, "../../models/target/test_model")
        assert os.path.isdir(modelFile)

        try:
            newModel: tm.KaggleModel = tm.load_model("test_model")
        except BaseException:
            print("Loading model failed.")
        finally:
            shutil.rmtree(modelFile)
