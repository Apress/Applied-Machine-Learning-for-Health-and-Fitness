import argparse
from azureml.core import Workspace, Model, Run

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, dest='model_name', default="activities", help='Model name.')
parser.add_argument('--model_folder', type=str, dest='model_folder', default="outputs", help='Model folder.')
parser.add_argument('--model_file', type=str, dest='model_file', default="activities.pkl", help='Model file.')
args = parser.parse_args()
model_name = args.model_name
model_folder = args.model_folder
model_file = args.model_file

run = Run.get_context()

print("Model folder:",model_folder)
print("Model file:",model_file)

Model.register(workspace=run.experiment.workspace,
               model_name = model_name,
               model_path = model_folder+"/"+model_file)

run.complete()
