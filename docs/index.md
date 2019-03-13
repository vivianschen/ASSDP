---
layout: default
title: Automatic Song Structure Detection Program
---
## A Study on Machine vs. Human Perception of Pop Song Structure

### feat. Karen Bao, Stacey Chao, Vivian Chen <br/>


## Intro 

Structure is one of the key characteristics that makes a song catchy by providing a balance between repetition and variation (despite the fact that we aren't always actively aware of it when we're listening to the song). Songs usually consist of a few different sections: Intro, Verse, Chorus, Bridge (or Transition), and Outro. Sometimes it is obvious where each section ends/begins and whether a section is the chorus, verse, etc.; other times...not so much. In this project, we focus on pop songs and whether machines can detect song structure better than humans can, especially when a song's structure is vague even to human ears. 


## How We Built It

We created our program by building off of two sources of code: Oriol Nieto's song segmenter ([https://github.com/urinieto/msaf](https://github.com/urinieto/msaf)) and Vivek Jayaram's chorus detector ([https://github.com/vivjay30/pychorus](https://github.com/vivjay30/pychorus)). The chorus detector originally determined the chorus based on repetition (the more it repeated, the more likely it was a chorus), but it turned out to fail for many songs. We improved it by incorporating [Librosa](https://librosa.github.io/librosa/)'s beat onset feature (the stronger the beat onsets a section had, the more likely it was a chorus). We then assigned the chorus to the segments from the segmenter that matched best to the chorus detected by the chorus detector. To determine the remaining segments (particularly verses), we used dynamic time warping ([https://pypi.org/project/fastdtw/](https://pypi.org/project/fastdtw/)) to calculate similarity measures between the segments. If the first or final segments were unique, they were assigned as "intro" or "outro", respectively.


## The Program

Our program returns a text file delimiting the time stamps of each segment of a song, which we uploaded into Audacity ([https://manual.audacityteam.org/man/creating_and_selecting_labels.html](https://manual.audacityteam.org/man/creating_and_selecting_labels.html)) to get the following: 


## User Testing

User testing was done on 9 subjects, all of whom are college students who listen to pop songs regularly. We used 4 genres of pop songs in different languages (American pop, Japanese pop, Korean pop, Chinese pop), with hard and easy songs each. Here are some examples of the results we had:


## Results
