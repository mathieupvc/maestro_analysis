import subprocess
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--data_folder", type=str, help="folder containing all the subjects of the maestro experiment")
parser.add_argument("-p", "--preprocessed_folder", type=str, help="destination folder that will have the same structure as the data_folder")
parser.add_argument("-d", "--done", nargs='+', type=str, help="subjects already processed (will not be processed again)", default="")
parser.add_argument("-w", "--wav_folder", type=str, help="folder containing the .wav stimuli (ordered by trial number)")
args = parser.parse_args()
data_folder = args.data_folder
preprocessed_folder = args.preprocessed_folder
wav_folder = args.wav_folder
done = args.done
print('data_folder:', data_folder)
print('preprocessed_folder:', preprocessed_folder)
print('Already done:', done)
print('wav_folder:', wav_folder)
print('')

path = Path(data_folder)
subject_folders = [x for x in path.iterdir() if x.is_dir() and x.name not in done]

for subject_path in sorted(subject_folders):
    sub_fol = list(subject_path.glob('**/*run01.fif'))[0].parent  # The .fif files are contained in a folder named with the date of recording
    fine_cal = list(subject_path.glob('**/*.dat'))[0]
    destination_path = Path(preprocessed_folder, subject_path.name)
    destination_path.mkdir(parents=True, exist_ok=True)  # Create the destination folder if it does not exist

    subprocess.run(["python", "meg_maestro_preprocessing.py",
                    "-s", str(sub_fol),
                    "-f", str(fine_cal),
                    "-ct", os.path.join(str(sub_fol), "sss_config", "ct_sparse.fif"),
                    "-b", "MEG0341",
                    #"-ch", "None",
                    "-d", str(destination_path),
                    "-w", wav_folder
                    ])
