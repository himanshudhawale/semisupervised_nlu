# Semi-supervised Natural Language Understanding

## Table of Contents

- [Semi-supervised Natural Language Understanding](#semi-supervised-natural-language-understanding)
  - [Table of Contents](#table-of-contents)
  - [Team](#team)
  - [Introduction](#introduction)
    - [**Semi-supervised** Natural Language Understanding:](#semi-supervised-natural-language-understanding)
    - [Semi-supervised **Natural Language Understanding**:](#semi-supervised-natural-language-understanding)
  - [Background](#background)
    - [Language Models](#language-models)
    - [Related work](#related-work)
  - [Problem Statement](#problem-statement)
    - [Motivation](#motivation)
    - [Dataset](#dataset)
    - [Approach and Summary of Methods](#approach-and-summary-of-methods)
  - [Logistics](#logistics)
    - [Updated Timeline](#updated-timeline)
    - [Updated Division of Work](#updated-division-of-work)
  - [Reflections](#reflections)
    - [Differences and novelty](#differences-and-novelty)
  - [References and Citation](#references-and-citation)

## Team

- Team Name: Regrayshun
- Team Members:
  - Dhruv Dhamani (ddhamani@uncc.edu)
  - Saloni Gupta (sgupta38@uncc.edu)
  - Himanshu Sunil Dhawale (hdhawale@uncc.edu)
  - Bhavya Chawla (bchawla@uncc.edu)

## Introduction

Let's break down the project title in two for clarity -

### **Semi-supervised** Natural Language Understanding:
  
Machine learning approaches are tend to either be supervised or unsupervised.

  - A supervised approach as in we when have curated data with a clear expectation for what parts of the data are the *input* to the model and what part the model is supposed to predict/infer or present as an *output*.
  - In general, unsupervised approaches mean that the data we have has not been labelled, classified or categorized. Instead, we create some sort of a feedback mechanism for the model which helps it identify commonalities in the data and react based on the presence or absence of such commonalities in each new piece of data.
  
  When we say *semi-supervised* what we mean is that we're going to take pre-existing labelled, classified or categorized data - the kind of data one uses for *supervised learning*, and transform and curate it into unlabelled data - the kind of data one uses for *unsupervised learning* - with the transformation or curation being done *on the basis of the labels of our pre-existing labelled data*, and then use this curated data for training our model.

  **Why would anyone do such a thing?** 
  
  Well, the motivation is simple. One could argue that as a general rule, there is a lot more unlabelled data in existence than labelled data. Thereby, if one creates a machine learning system that learns by use of unlabelled data, it is always going to have more data to learn from than a system that is based on learning from labelled data.

  And in most cases, the more data you have, the better machine learning systems can learn. Thusly, by transforming and curating labelled data for supervised learning approaches into unlabelled data for unsupervised learning approaches, we also manage to increase the available data for learning manifold; assuming the area of application of said machine learning system have availability of unlabelled data that can be learned from, and satisfactory feedback mechanisms for unsupervised learning to take place.

### Semi-supervised **Natural Language Understanding**:
  
  Natural language understanding or Natural language inference is a subtopic of natural-language processing in artificial intelligence that deals with machine reading comprehension. It can be thought of as what happens after natural language processing (NLP) - if a computer was presented with the sentence -
    
    This is a sentence.

  The results of performing NLP techniques and algorithms on this sentence would give us information about what the individual words in the sentence are relative to the grammar of that language, and what the relationship between the words in the sentence is. It would look something like -

  ![Example of spacy's NLP](spacy-viz.svg)

  Now taking this set of information produced by NLP algorithms and getting a machine to _comprehend_ or _understand_ what the sentence means is what Natural Language Understanding/Inference is.

  Our above discussion of the motivation behind doing "semi-supervised" learning lead us to the conclusion that semi-supervised learning might be a good thing to do in the following scenarios -

  - when there is availability of large amounts of raw data that can be learned from
  - when there exist satisfactory feedback mechanisms to facilitate unsupervised learning
  - (and also when labelled data can be transformed and curated into labelled data without a whole lot of hiccups)

  One field that satisfies all of these requirements is Natural Language processing -

  1. Availability of large amounts of raw data - Large amounts of written down, digital representations of language, i.e. text, is basically what the internet is. I don't think more needs to be said here.
  2. Existence of satisfactory feedback mechanisms to facilitate unsupervised learning - Academia has been doing unsupervised learning in the field of Natural Language Processing for years. The process of capturing meaning of words in a vector of floating point numbers by simply providing lots of examples of how the words are used in the natural language *is* unsupervised learning.
  3. Ease of transformation and curation of labelled data into unlabelled data - more on this in the review of related work and the approach and summary of methods sections below.

## Background

### Language Models

### Related work

## Problem Statement

### Motivation

### Dataset

### Approach and Summary of Methods

## Logistics

### Updated Timeline

### Updated Division of Work

## Reflections

### Differences and novelty

## References and Citation


<h2 id="fine-tuning-language-models-to-solve-natural-language-understanding-tasks"><strong>Fine-tuning Language models to solve Natural Language Understanding tasks</strong></h2>
<p><strong>Team Name:</strong> Regrayshun</p>
<p><strong>Team Members</strong><br>
Bhavya Chawla(801081909)<br>
Dhruv Dhamani(801084292)<br>
Himanshu Dhawale(801084142)<br>
Saloni Gupta(801080992)</p>
<p><strong>Introduction</strong><br>
In everyday life we use Google’s <a href="http://API.ai">API.ai</a>, Amazon’s Alexa, Microsoft’s <a href="http://Luis.ai">Luis.ai</a> and other artificially intelligent assistants that are trained to understand our language and process it to perform tasks. So, taking an inspiration from these assistants, we are aiming to fine tune a semi-supervised natural language model that is capable of understanding human language without any prior knowledge. The model we will be using to fine tune is the GPT-2 vanilla model which is a transformer-based language model with 1.5 billion parameters and serves the purpose of simply predicting the next word.It comsists of a broad set of capabilities like generating synthetic text samples of unprecedented quality. A benefit that the model serves is that it does not require domain specific knowledge to predict the words, it simply learns from raw text using no task-specific training datasets.</p>
<p>Omni-Supervised learning was defined as a special regime of semi-supervised learning in which the learner exploits all available labeled data plus internet-scale sources of unlabeled data in a paper by Facebook AI Research (FAIR) in the 2017 paper Data Distillation: Towards Omni-Supervised Learning . A fancy name to give to something researchers in NLP have been doing for years, word embeddings have been sourced from internet-scale data, and then applied to several tasks achieving state-of-the-art results.</p>
<p>While the researchers at OpenAI made no attempts at fine tuning the GPT2 on various tasks - the whole point of the paper was that language models trained with quality data can achieve competitive results on various tasks without any fine tuning. However, we couldn’t help but be very excited about finding out how such a model would perform with fine tuning, considering that, to the best of our knowledge, there has never been any language model trained with data of this quality, and scale without the data being bastardized by any harsh pre-processing. So, that is what we plan on doing. Utilizing the smallest pretrained GPT2 model released by OpenAI, we would be fine tuning the model and evaluating its performance on either the open, crowd-sourced NLU benchmark by <a href="http://Snips.ai">Snips.ai</a>, or the NLU Evaluation Corpora (Braun et al.), whichever proves to be easier to work with. We’ll also be experimenting with the use of data augmentation in the question-context-answer format proposed by McCann et al., by paraphrasing the questions and answers, which we hypothesize will result in a model that generalizes better.</p>
<p><strong>The Problem</strong><br>
Whenever one person asks another person a question, it is important for the listener to extract the query from the sentence. Similarly for a language processing model to answer a question, the most important task is to extract the query that the user is making. In the world of Natural Language Processing, this problem is referred to as slot filling. As our language model uses no prior knowledge about the context, it aims to fill these slots with the correct answer after understanding the query asked. For example, if the user asks, ‘Which is the most famous restaurant in the city?’, the language model should be capable enough to understand the grammar of the sentence to identify the important aspects. We will be evaluating the performance based on the metrics we learned in the class, precision and recall. Precision measures how exact are the attributes extracted and recall measures the amount of existing attributes that are recovered by the model.</p>
<p><strong>Dataset</strong><br>
The dataset that we are using for our project has been sourced from a company named Snips. We are dealing with seven intents namely, AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork and SearchScreeningEvent. Each intent comprises of more than 2000 queries that have been generated with crowdsourced method.</p>
<p><strong>Model</strong><br>
Explain the fine tuned model and how it works.</p>
<p><strong>Literature Survey</strong><br>
Data Distillation: Towards Omni-Supervised Learning]
In this paper the researchers have investigated omni-supervised learning and proposed the concept of data distillation, a method capable of ensembling predictions from multiple transformations of unlabeled data, using a single model to automatically generate new training annotations. They were able to experiment and prove that their model could surpass large scale supervised learning with the help of data distillation.</p>
[The Natural Language Decathlon: Multitask Learning as Question Answering]
The focus of this paper was on introducing a new benchmark for measuring the performance of NLP models. They presented MQAN model for simple question answering which capitalizes on questions with the help of a multi-pointer-generator decoder. They demonstrated how labelled data can be used to train a language model to perform multiple tasks by casting all tasks as question-answers over a context. We are using this concept to understand the question answer mechanism in order to make the model extract the correct query to perform slot filling.</p>
[Language Models are Unsupervised Multitask Learners]
This paper has been the major inspiration for our project as it focuses on all the correct aspects of training a language model to develop unsupervised learners. This paper by researchers at OpenAI has shown how better quality data, and a more complex transformer based architecture results in a model that can achieve state-of-the-art results without any fine tuning whatsoever. In order to achieve better quality of data, they scraped web pages which had been curated by humans, especially all the outbound links from Reddit, a social media platform, which performed as a heuristic indicator to identify if the users found the link to be interesting, educational or funny. Their model operates on a byte level and does not require lossy pre-processing or tokenization. We will be using this paper as our basis for fine tuning the model to have a more efficient and effective language prediction model.</p>
[Exploring the Limits of Language Modeling]
This research dedicatedly explored different techniques like character convolutional network, Long-Short term memory, etc to deal with challenges like corpora and vocabulary sizes and complex long term structure of language. They demonstrated how the state-of-art model performed significantly better using lesser number of parameters. They scaled RNN based language models on 1 billion word benchmark to outperform competing models including tuned N-grams. This study helped us understand the state-of-art model better and how language models can be further improved using correct softmax approximations with important sampling. We were also able to extend our understanding of the regularization concept much deeply.</p>
At least 10 examples</p>
<p><strong>Method</strong><br>
Describe how are you fine tuning it.</p>
<p><strong>Updated Plan</strong></p>
<p><strong>Reflection of Feedback</strong></p>
<p><strong>What we have achieved</strong></p>
<p><strong>Future Scope</strong></p>
<p><img src="https://lh4.googleusercontent.com/BfpEeDUIgI-IRx-5QInO0JNBpB5_eQkDwIIq4jszusSaFI6UfWVPQGo8HhGywcZUdC5avxdYAYpzMPxUdiE5EJqHl_H8RV-A5EhUMu70lHqTDk0ffD9n0OhB_8m3eL3hghyB3oMV" alt=""></p>

