import attack_model as am
import os.path
import shutil
import os


class TestAttackModel():
    def test_load_save_model(self):
        targetModel: am.KaggleAttackModel = am.KaggleAttackModel(100)
        am.save_model("test_model", targetModel)
        curDir = (os.path.dirname(__file__))
        modelFile: str = os.path.join(curDir, "../../models/attack/test_model")
        assert os.path.isdir(modelFile)

        try:
            newModel: am.KaggleAttackModel = am.load_model("test_model")
        except BaseException:
            print("Loading model failed.")
        finally:
            shutil.rmtree(modelFile)
