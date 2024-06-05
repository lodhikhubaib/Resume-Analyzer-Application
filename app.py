import pickle
import re
import nltk
import streamlit as st
nltk.download('punkt')
nltk.download('stopwords')

tfidf = pickle.load(open('vector.pkl','rb'))
knn = pickle.load(open('model.pkl','rb'))

def remove_irrelvent_data(txt):
  txt_clean = re.sub('http\S+\s',' ',txt)
  txt_clean = re.sub('@\S+',' ',txt_clean)
  txt_clean = re.sub('#\S+',' ',txt_clean)
  txt_clean = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt_clean)
  txt_clean = re.sub('\s+', ' ', txt_clean)
  txt_clean = re.sub(r'[^\x00-\x7f]',r' ',txt_clean)
  txt_clean = re.sub('RT | cc',' ',txt_clean)
  return txt_clean

def main():
    st.set_page_config(page_title="Resume Analyzer App", page_icon="logo.png")
    st.title("Resume Analyzer App")
    uploaded_file = st.file_uploader('Upload Your Resume',type = ['txt'])
    if uploaded_file is not None:
        try:
            resume_byte = uploaded_file.read()
            resume_txt = resume_byte.decode('utf-8')
        except UnicodeDecodeError:
            resume_txt = resume_byte.decode('latin-1') 
            
        clean_resume = remove_irrelvent_data(resume_txt)
        input_features = tfidf.transform([clean_resume])
        pred = knn.predict(input_features)[0]
        # st.write(pred)
        
        category_map = {
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
        category = category_map.get(pred,'unknown')
        st.write("Predicted Category is: ",category)
if __name__ == '__main__':
    main()