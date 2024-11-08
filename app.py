import numpy
import nltk
import regex
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
# tfidf = TfidfTransformer(stop_words = 'english')



#loading models

clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

# web app

def Clean_Resume(txt):
    # Use a regex pattern to match URLs (with or without trailing whitespace)
    cleanTxt = regex.sub(r'http\S+', '', txt)
    cleanTxt = regex.sub(r'RT|CC', '', cleanTxt)
    cleanTxt = regex.sub(r'@\S+', '', cleanTxt) # --> @
    cleanTxt = regex.sub(r'#\S+', '', cleanTxt) #--> hastags
    cleanTxt = regex.sub(r'[%s]'% regex.escape("""!"#$%^&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', cleanTxt) #-- > special char
    cleanTxt = regex.sub(r'[^\x00-\x7f]', '', cleanTxt) # -- > 
    cleanTxt = regex.sub(r'\s+', ' ', cleanTxt) # --> \n ,\t

    return cleanTxt


def main():
    st.title(" Job Matching using Resume ")
    
    
    upload_file=st.file_uploader('Upload Resume',type=['txt','pdf'])
    category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

    
     # Template Resumes
    st.subheader("Or select a template resume:")
    template_options = {
        "Java Developer": "templates/java_developer.txt",
        "Data Scientist": "templates/data_scientist.txt",
        "HR Specialist": "templates/hr_specialist.txt",
        "Business Analyst": "templates/business_analyst.txt"
    }
    
    # Display template options
    selected_template = st.selectbox("Choose a template", options=list(template_options.keys()))
    if st.button("Check Template Resume"):
        # Load and process the selected template
        with open(template_options[selected_template], 'r') as file:
            resume_text = file.read()
            cleaned_resume = Clean_Resume(resume_text)
            cleaned_resume = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(cleaned_resume)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
            st.write(f"Predicted Category: {category_name}")
    
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_txt = resume_bytes.decode('utf-8')
        
        except UnicodeDecodeError:
            
            resume_text = resume_bytes.decode('latin-1')


        cleaned_resume=Clean_Resume(resume_text)
        
        cleaned_resume= tfidf.transform([cleaned_resume])
        
        prediction_id = clf.predict(cleaned_resume)[0]
        
        
        
        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write(category_name)
        print("Predicted Category:", category_name)
        print(prediction_id)
        

#python main 

if __name__ == "__main__":
    main()