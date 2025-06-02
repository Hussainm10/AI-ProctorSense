# AI-ProctorSense
ProctorSense is an AI-powered motion detection tool designed for educational institutions to enhance security during online exams, lab sessions, and restricted area monitoring.


 ProctorSense – Motion Detection Tool for Educational Security

**ProctorSense** is a smart motion detection system designed specifically for inspection and security in educational environments. Leveraging computer vision and deep learning, ProctorSense can effectively detect suspicious or unauthorized motion during critical scenarios such as online examinations, lab sessions, and restricted access zones.

Project Overview

In today's digitally transforming educational sector, ensuring fairness and discipline during online and physical assessments has become more critical than ever. ProctorSense addresses this by acting as a virtual invigilator that monitors for motion-based anomalies and raises real-time alerts to flag them.


Team Behind ProctorSense

**Team Lead & Developer**   | Hussain Mansoor Bhutto    
**Co-Lead & Graphics Designer** | Muhammad Moosa Khan        
**Junior Developers**     | Mahnoor Noman, Muneera Quaid & Karishma Kumari          

Our team collaboratively worked to design, code, and present a functional prototype of ProctorSense tailored for institutional settings.

---

Tools & Technologies

**Python 3.x**
**OpenCV** for motion detection
**Streamlit** (GUI support)
**D-library** and **MediaPipe** for facial landmarks detection

**How to Set Up and Run ProctorSense**
Follow these steps to successfully run ProctorSense, our AI-powered motion detection and inspection tool.

Requirements
Make sure your system has the following:
Python 3.7 or higher
pip (Python package installer)
A webcam (for real-time video detection)

**1.Clone the Repository**
git clone https://github.com/yourusername/proctorsense.git
cd proctorsense

**2.Create and Activate a Virtual Environment (Optional but Recommended)**
python -m venv venv
# Windows
venv\Scripts\activate

**3.Install Required Packages**
Use the following command to install all dependencies:
pip install -r requirements.txt
NOTE : If dlib fails to install using pip, use the .whl file method below.

Install dlib Using a .whl File (For Windows Users)
Download the compatible .whl file for your Python version from this site:

**https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib*
pip install dlib‑19.24.0‑cp38‑cp38‑win_amd64.whl

**4.Download the Dlib Facial Landmark Model File**
ProctorSense uses a pre-trained model for facial landmark detection.
www.dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

After downloading:
Extract the .bz2 file
Place the shape_predictor_68_face_landmarks.dat file into the project directory (same folder as the main script)

**5.Run the ProctorSense Tool**
Once everything is set up, simply run the main script in terminal :
python proctorsense.py







