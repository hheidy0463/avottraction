# 🥑 Avottraction - Codeology Sp25

Avottraction uses a modern approach to detect “attraction signals” in text through advanced natural language processing. First, the text is cleaned by removing special characters, expanding contractions, and more so that only meaningful information remains. Then, we fine-tune a DistilBERT model to learn subtle context. By splitting the data carefully, we can accurately measure performance and avoid overfitting. Using the Hugging Face Trainer framework simplifies training and evaluation, and early stopping helps us quit when the model stops improving. Overall, Avottraction is flexible enough to adapt to different fields or integrate into chat systems, showing practical ways to apply text classification.

## 🗒️ Overview 

- Tech Areas: Machine Learning, Web Development

- Tools / Technologies:
  - Machine Learning Model: DistilBERT
  - Python
  - Hugging Face Transformers
  - scikit-learn
  - PyTorch, NLTK, and regex for preprocessing
  - pandas and numpy for data manipulation
  - Web Dev: HTML, CSS
  - Data: OpenAI API, Reddit PRAW

## 👫 Project Members

Project Manager: Miller Liu

Project Leaders: Vivek and Heidy

Project Members: Sanghun, Brooke, Sajiv, Karen, Shriyaa

## 💻 Set Up
***[Avottraction](https://aquamarine-handbell-a5f.notion.site/Avottraction-16e172c5e5608099acc1c790545f560d?pvs=4)***

This notion has everything you’ll ever need!
The calendar is the main resource, check it frequently. We will communicate through Facebook Messenger or iMessage.


<details>
  <summary>Cloning and Repo Setup</summary>
  <br>

1. Create a new repo on GitHub

2. Clone our skeleton code to your local machine:

   ```bash
   git clone <PROJECT URL HERE>
   ```

3. Set the remote origin to be YOUR newly created repo (this is so you can make commits to your own repo on GitHub):

   ```bash
   git remote remove origin
   git remote set-url origin <your newly made GitHub repo url>
   ```

4. Set the remote “start origin” to be OUR skeleton code repo (this is so you can get updates to our starter code):

   ```bash
   git remote add starter <PROJECT URL HERE>
   ```

5. Now you can get the latest starter code with the following command:

   ```bash
   git pull starter main
   ```

6. Send Vivek and Heidy the link to your repo via iMessage

</details>



<details>
  <summary>Installation/Dependencies</summary>
  <br>

1. Install via terminal:

   ```bash
   pip install -r requirements.txt
   pip install nltk
   ```
2. Run the following (in a Python shell or script):

   ```bash
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
  ``` </details>
