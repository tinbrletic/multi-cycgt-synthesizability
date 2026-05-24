"""Verify the dgllife monkey-patch is in place: GCNPredictor.forward accepts model_use kwarg."""
import inspect

from dgllife.model.model_zoo.gcn_predictor import GCNPredictor

src = inspect.getsource(GCNPredictor.forward)
if "model_use" in src:
    print("PATCH OK")
else:
    print("NOT PATCHED -- re-run the Copy-Item step from SETUP_AND_RUN.md")
