## Dataset

- Images (`screens/env_name/episode_num/filename.npy`): 84x84 in Gray-scale
- Actions (`trajectories/env_name/summary.txt`): (num_of_episode, frame, reward, score, terminal, action)

### Directory Structure

- Total frame: 9,679,856
- Whole data size: 97.7 GB
- Directory Architecture: 

```shell
/atari_v2_release/
├── screens # frames of game scenes
│    ├── mspacman
│    ├── pinball
│    ├── qbert
│    ├── revenge
│    └── spaceinvader
└── trajectories # (num_of_episode,frame,reward,score,terminal,action)
     ├── mspacman
     ├── pinball
     ├── qbert
     ├── revenge
     └── spaceinvader

# Data Storage of each folder in `trajectories`
$ du -sh ./trajectories
36M	./pinball
50M	./qbert
43M	./mspacman
66M	./revenge
58M	./spaceinvaders
252M	.

# Data Storage of each folder in `screens`
$ du -sh ./screens
not working.... maybe way toooo huge
```

### Summary of Dataset

- Overview

  I don't know why but mostly the volume has been increased

![data_description](/home/noio0925/Desktop/atari_v2_release/images/data_description.png)

- Distributions

  ![distribution](/home/noio0925/Desktop/atari_v2_release/images/distribution.png)



## Data Preprocessing

Since the game was recorded by images of the screenshots using `png`, the data size was huge. So I first decided to preprocess them before moving on to the training the model with them.

1. Downsize the image from 210x160 to 84x84
2. Turn RGB image into Gray-scale
3. Normalise the image into the range between 0 and 1 by dividing the pixel by its maximum value, which is normally 255
4. Dealing with all frames is computationally expensive and not feasible so that assuming that the agent takes the same actions over pre-defined `K` timesteps(normally k = 4), I saved the image once in K timesteps
5. Although I skipped some timesteps, there was some features only appear in certain frames. So to retain them in the state space, which is in this case images, I have merged the images of the last two steps over `K` timesteps. Therefore, I can safely preserve important features in images.



## Reference

- Paper: The Atari Grand Challenge Dataset: https://arxiv.org/pdf/1705.10998.pdf
- Project Page: http://atarigrandchallenge.com/data
- Games: http://atarigrandchallenge.com/