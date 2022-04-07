# Intent-Based-Chatbot

## Table Of Content

- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [To Do](#to-do)

## Overview

The project happens to be an AI Intent-based Chatbot, a custom dataset is used to train the model utilized. The model type comes under supervised learning & a sub-type Classification.

## Motivation

Customer support is crucial in any functioning of any business, thus more importance should be given to entertaining any necessary query of the customer. But when we run the business, it's extremely important how we segregate the manpower needed, thus the energy given to a specific task. An AI Intent-based Chatbot, therefore, helps in addressing simple queries of the customer. Consequently, saving the manpower which can be used for other tasks.


## Technical Aspect 

The Chatbot designed explicitly addresses the queries of the customer for a restaurant, as the dataset created was relevant to that field. The dataset has 10 tags along with 5-6 questions about the relevant tag.

The libraries utilized are **TensorFlow, Keras & NumPy**. First, the question asked is passed through a preprocessing pipeline, which converts the capital letters to lower-case, and also, Lancaster stemmer is utilized for the task of converting the word to its stem word. Both tasks thus reduce the vocabulary size & help in reducing the computation time and increasing the efficiency of the model. 

The classification task involves the processing of the question and then based on the question prediction of the tag. Based, on the tag the algorithm returns a randomly selected answer from the answer pool. 

![](https://github.com/gauravshipurkar/Intent-Based-Chatbot/blob/main/Result.png)

## Installation

The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/release/python-380). To install **Open-CV** you can go [here](https://opencv.org/). To install the required packages and libraries, run this command in the project directory after cloning the repository:

```
pip install -r requirements.txt

```
## To Do

- Front-end for the project
