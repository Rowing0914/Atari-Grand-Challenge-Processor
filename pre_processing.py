import os, collections
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd


GAME_NAMES  = ["mspacman", "pinball", "qbert", "revenge", "spaceinvaders"]
INPUT_SHAPE = (84,84)
SKIP_FRAME  = 4

def _trajectory_processing():
	for GAME_NAME in GAME_NAMES:
		print("===== PROCESS ", GAME_NAME)
		temp = list()
		for traj in os.listdir('./trajectories/' + GAME_NAME):
			f = open('./trajectories/' + GAME_NAME + '/' + traj, 'r')
			# skip first row!
			next(f)
			# skip sencond row as well!
			next(f)
			for line in f:
				_temp = line.strip().split(",")
				# [num of episode, frame, reward, score, terminal, action]
				temp.append([int(traj.replace(".txt", "")), int(_temp[0]), int(_temp[1]), int(_temp[2]), int(_temp[3]), int(_temp[4])])
		print(len(temp))
		df = pd.DataFrame(temp)
		del temp
		df.columns = ["num_of_episode", "frame", "reward", "score", "terminal", "action"]

		# remove unneccesary files in the directory
		os.system("rm {}".format('./trajectories/' + GAME_NAME + '/*'))
		df.to_csv('./trajectories/' + GAME_NAME + '/summary.txt', index=False)
		del df

def _screens_processing():
	_obs_buffer = collections.deque(maxlen=2)
	for GAME_NAME in GAME_NAMES:
		for _episode_index in os.listdir('./screens/' + GAME_NAME):
			frame_cnt = 0
			processed_images = list()
			print("===== PROCESS {0}: {1}".format(GAME_NAME, './screens/' + GAME_NAME + '/' + _episode_index))

			with tqdm(os.listdir('./screens/' + GAME_NAME + '/' + _episode_index)) as image_files_in_episode:
				# process the image step by step in a specific episode
				for index, _image in enumerate(image_files_in_episode):
					image_files_in_episode.set_postfix(collections.OrderedDict(current_file=index))
					_obs_buffer.append(_preprocess('./screens/' + GAME_NAME + '/' + _episode_index + '/' + _image))
					
					# take the maximum feature of the image over two steps
					iamge_matrix = np.max(np.stack(_obs_buffer), axis=0)

					if frame_cnt == SKIP_FRAME:
						# normalise the pixels
						iamge_matrix = np.array(iamge_matrix).astype(np.float32) / 255.0
						# save image
						processed_images.append(iamge_matrix)
						# clear buffer
						_obs_buffer.clear()
						frame_cnt = 0
					frame_cnt += 1

			np.save('./screens/' + GAME_NAME + '/' + _episode_index + '/images', np.array(processed_images))
			os.system("rm {}".format('./screens/' + GAME_NAME + '/' + _episode_index + '/*.png'))


def _preprocess(file):
	"""
	1. Turning an image into gray-scale
	2. Downsize the image to INPUT_SHAPE

	Args:
		file: an image file

	Returns:
		a numpy array of processed image
	"""
	img = Image.open(file)
	img = img.resize(INPUT_SHAPE)
	img = img.convert('L')
	# img.save('greyscale.png')
	return np.array(img)

if __name__ == '__main__':
	# _screens_processing()
	# _trajectory_processing()