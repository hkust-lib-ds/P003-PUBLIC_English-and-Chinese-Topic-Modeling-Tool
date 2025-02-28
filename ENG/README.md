# Manual Guide for using our English Topic Modeling Tool

## Introduction

**Topic modeling** is a technique in natural language processing (NLP) and machine learning that is used to discover abstract topics within a collection of documents. It helps in identifying the underlying themes present in a large corpus of text by analyzing the patterns of word usage.

Our tool offers two main functionalities:
- **Train Model with Training Data**
- **Use Model without Training Data**

This manual will guide you through these. Hope you enjoy your journey using this tool!

There are three sections in this manual:
+ [Usage Guidance](#tool-overview)
+ [Common Issues](#common-issues)
+ [A Sample Use Case](#sample-use-case)

## Download and Use

Please follow the steps in this page for installation: https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool

## Tool Overview
The tool comprises **two** main components: **Sidebar** and **Main Page**, with the following functionalities:
![Overview of the tool](manual-img/0_Overview.png)

### Quick links
+ **Sidebar**
    + Define Model
        + [Upload model with training data](#usage-1-train-a-topic-model-with-training-data)
        + [Upload model without training data](#usage-2-use-a-topic-model-without-training-data)
        + [Train model](#option-2-train-model)
    + View Topic
        + Display filtering for: [discovered topics](#displaying-filtering); [prediction & topic changes over time](#6-display-filtering-for-prediction-and-topic-changes-over-time)
        + [Search topics](#search-for-topics-with-a-keyword)
    + [Update Topic](#update-topics)
+ **Main Page**
    + [Import Training Documents](#1-import-training-data)
    + Discovered Topics
        + [Visualization & details](#visualization--details)
        + [Export](#3-data-export-of-the-model)
    + [Apply the Model to New Data](#4-apply-the-model-to-new-documents)
    + [Track Topic changes Over Time](#5-track-topic-changes-over-time)


---

> _In the usage example below, the sample data comes from our HKUST Library Special Collection's [Paul T. K. Lin’s Hsinhua Photo Collection](https://lbezone.hkust.edu.hk/rse/paul-tk-lin-hsinhua-photo-collection)._


## Usage 1: Train a Topic Model with Training Data
You can use our tool to train a model for topic modeling. A tailored model enhances the relevance and accuracy of topic identification, adapting to the unique context of your data. Especially, it can be reused as your data expands over time, ensuring consistent performance and deeper insights.

Steps as follow:
![Overview of usage 1](manual-img/1-00_Usage_1_Overview.png)

### (1) Import Training Data
Go to the **"Upload Documents"** section in the main page.

#### Avaliable File Types
**3** types of files are supported for uploading: 
- **TXT**
- **CSV**
- **Mixed TXT & CSV**

Please click the **"Guidance for File Format"** button in the tool for more details and download sample files.

#### Time Format
If you wish to use the function **"Track Topic Changes Over Time"** later, you will need to provide **time information** together with the uploaded data, and you must use **CSV** files. If you do not wish to use this function, please choose **"No Time Format"**.

![Import training data](manual-img/1-01_Import_Training_Data.png)

### (2) Define Model
Go to the **"Define Model"** section in the sidebar.

#### Option 1: Upload Model
You can upload a previously saved `.pickle`  format file. **<span style="color:red">Attention: you must use the same environment as the one used when saving the pickle file.</span>**

#### Option 2: Train Model
Alternatively, you can set the **parameters** and the **stopwords list** (optional, either default or customized) to train a new model based on the documents you uploaded. 

_You may click the **"Info Icon"** next to each parameter for more details._

![Define model](manual-img/1-02_Define_Model.png)

### (3) Analyze Discovered Topics
Now we are ready to analyze the topics we discovered! 
Go to the **"Discovered Topics"** section in the main page.

#### Visualization & Details
Various figures are provided to assist your analysis in the **"Visualization"** section. Click on each tab to explore! You can also **download** a figure/image by clicking the corresponding button at the top right corner (if available). 

If you want to have a deeper understanding, you can scroll down to view the **"Details"** section. 

The following summarizes the information provided in this section:
+ Topic:
    + Topic name
    + Topic label (if any)
    + Representative words
+ Documents:
    + Document ID
    + Affiliated topic with the corresponding probability
    + Content with representative words highlighted

Please view **"Remark for Display"** in the tool for more details.

![Visualization & details](manual-img/1-03_Analyze_Discovered_Topics.png)

#### Displaying Filtering
You can filter the above information by choosing **ALL TOPICS** or **a specific topic** in the sidebar **"View Topic"** section.
##### All Topics
![When choosing all topics](manual-img/1-04_All_Topics1.png)
![When choosing all topics](manual-img/1-05_All_Topics2.png)
##### One Topic
![When choosing one topic](manual-img/1-06_One_Topic.png)

#### Search for Topics with a Keyword
You can search for a certain number of topics with a specified word in the **"View Topic"** section in the sidebar.

![Search topics](manual-img/1-07_Search_Topics.png)

#### Update Topics
After the above analysis, you may notice something interesting or incorrect. You can update the topics accordingly in the **"Update Topic"** section in the sidebar, or [re-train](#2-define-model) the model with another setting in the **"Define Model"** section in the sidebar.

You can update the generated topics in 3 ways:
- **Label Topics**: Assign a name for the topic
- **Reduce Topics**: Reduce the total number of topics
- **Merge Topics**: Merge multiple specific topics into one topic

_Please click the **"Info Icon"** next to each operation for more details._

![Update topics](manual-img/1-08_Update_Topics.png)

### (3) Data Export of the Model
When you are happy with the result, you can **export the data** and **download the model** in the **"Discovered Topics"** section in main page. 

Both CSV and JSON format are provided in the exported zip file for some of the files. 

![Model data export](manual-img/1-09_Model_Data_Export.png)

### (4) Apply the Model to New Documents
Go to **"Apply the Model to New Documents" section** in the main page. This method is similar to training documents without time format. The results can be exported as well.

![Prediction](manual-img/1-10_Prediction.png)

### (5) Track Topic Changes Over Time
If there is a column of [time information](#time-format) in your uploaded CSV file, you will see the **"Track Topic Changes Over Time"** section in the main page.

#### Generate Timestamp
If your time format is **"Year_Month_Day"** or **"Customized"**, you will first need to generate timestamps. Choose an appropriate number of clusters according to the scatter plot. 

#### Find Changes over Time & Data Export
You will see the analysis results with visualization (summary & individual details) after clicking the **"Find Changes over Time"** button. The results can also be exported.

![Topic over time](manual-img/1-11_Topic_Over_Time.png)

### (6) Display filtering for Prediction and Topic Changes over Time
The display filtering function can also be applied for these two sections.

![Displaying filtering](manual-img/1-12_Displaying_Filtering.png)


## Usage 2: Use a Topic Model Without Training Data
If you have previously trained a topic model (a pickle file), you can use our tool for inference only by following steps below:

![Overview of usage 2](manual-img/2-00_usage_2_Overview.png)

### (1) Upload Model
Importing training data will not be needed for this usage. All you need is to upload the model file (in .pickle format).

![Upload Model](manual-img/2-01_Upload_Model.png)

### Analysis & Export & Prediction
The following are functions avaliable in this usage. 

![Functions avaliable](manual-img/2-02_Functions_Availiable.png)

## Common Issues
Here are some common issues and their corresponding solutions.

### (1) Unsatifactory Result
There is a certain degree of **randomness** during the model training. The same training data may result in different outcomes. If you get unexpected results (e.g. fewer than 5 topics), please try re-training or adjusting the training parameters.

### (2) Long Refresh Time
The refresh time is proportional to the amount of display content, which is closely related to the amount of training data. Therefore, it is suggested to separate the training process and prediction process. This means you can train a model following [usage 1](#usage-1-train-a-topic-model-with-training-data), download the model, and then perform prediction following [usage 2](#usage-2-use-a-topic-model-without-training-data).

## Sample Use Case
Here is a simple use case to help you better grasp how you can use our tool to analyse your data.
![Use Case Overview](manual-img/3-00_Use_Case_Overview.png)

### (1) Data
800 training data samples and 20 prediction data samples are randomly selected from this Kaggle dataset ["Topic Modeling Airline Reviews with BERTopic"](https://www.kaggle.com/code/gvyshnya/topic-modeling-airline-reviews-with-bertopic/).

### (2) Import & Training
Import the training documents and specify the time format. Train a model with the appropriate parameters. 
![Import & Training](manual-img/3-01_Import_Training.png)

![Set parameters](manual-img/3-01_Cropped_1-02.png)

### (3) Analyze & Update Discovered Topics
After the initial training, we can analyze the results using various visualizations. In this case, according to the representative words and the distance between the two clusters, we can notice that Topic 6 and Topic 8 are both negative reviews, although from different aspects. It seems reasonable to merge them.
![Analyze & Update](manual-img/3-02_Analyze_Update.png)

With a satisfactory result after merging, we can now assign labels with the help of the visualizations. For example, according to the bar charts of word scores, we can notice that the first few words of some topics (e.g. Topic 2) are dominant, which can be directly used as the label, while the importance is more distributed in other topics (e.g. Topic 0), so labels may need to be summarized.
![Label Topic](manual-img/3-03_Label_Topic.png)

Now we are ready to view the final result!
![View Result](manual-img/3-04_View_Result.png)

### (4) Topic Changes over Time
We can generate 20 timestamps and analyze topic changes over time.
![Topic Changes over time](manual-img/3-05_Topic_Over_Time.png)

According to the individual bar charts, we notice that the frequencies of some topics are increasing while others remain relatively stable.
![Topic Changes over time](manual-img/3-06_Topic_Over_Time_Result.png)

### (5) Apply the Model to New Data
In order to shorten the refresh time during prediction, we should download and re-upload the model following the steps that mentioned in [usage 2](#usage-2-use-a-topic-model-without-training-data).
![Download & Upload](manual-img/3-07_Download_Upload.png)

Now we can upload new data to perform prediction so as to find the topic distribution among the new data.
![Prediction](manual-img/3-08_Prediction.png)


