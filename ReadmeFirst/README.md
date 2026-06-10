# Doc-Intelligence-PID
Doc-Intelligence-For-Personal-Insurance-Domain
(Has been developed and Tested on Windows 11 & Ubuntu 24)

NOTE 01: All are LOCAL Setup/ Deployments and RUN Locally
NOTE 02: the environment configs are saved under "./config/" folder

EXPECTATION: Believe that you are a AIML + GenAI Engineer/ Developer with experience,
at least in handling LLMs/ vLLMs/ mLLMs along with Python, NodeJS, JavaScript and PATH,
also Believe that you are comfortable in understanding and using and tweaking the PATH,
in the code and comfortable in creating Conda/ MiniConda environments and have adequate,
or sufficient knowledge in Enterprise Systems including FastAPI


=====================================================================================================
# For Qwen 2.5 vLLM with Computer Vision
=====================================================================================================
01. Please Create a Hugging Face CLI Environment
02. Create a Folder Structure "D:\Qwen\Qwen2.5-VL-7B-Instruct\" for Windows 11 or
    "~/Qwen/Qwen2.5-VL-7B-Instruct/" for Ubuntu 24
    and download the model using Hugging Face CLI
=====================================================================================================

=====================================================================================================
# For Agentic AI Mini Language Model (Sentence Transformer for Embeddings)
=====================================================================================================
01. Please use the created Hugging Face CLI Environment
03. Create a Folder Structure "D:\sentence-transformers\all-MiniLM-L6-v2\" for Windows 11 or
    "~/sentence-transformers/all-MiniLM-L6-v2/" for Ubuntu 24
    and download the model using Hugging Face CLI
=====================================================================================================


=====================================================================================================
# Backend Context Root
=====================================================================================================
Context Root for Backend in Windows:
D:\PyCharmWorkSpace\Doc-Intelligence-PID\

Context Root for Backend in Ubuntu 24:
~/PyCharmWorkSpace/Doc-Intelligence-PID/
=====================================================================================================

=====================================================================================================
# Frontend Context Root
=====================================================================================================
Context Root for Frontend in Windows:
D:\PyCharmWorkSpace\Doc-Intelligence-PID\react-app\policy-ui\

Context Root for Frontend in Ubuntu 24:
~/PyCharmWorkSpace/Doc-Intelligence-PID/react-app/policy-ui/
=====================================================================================================

=====================================================================================================
# For DATA, FAISS-DB, Running Backend
=====================================================================================================
01. Please Configure Conda Environment First for:
    (a). Python 3.11 for Backend
    (b). NodeJS 18 for Frontend

02. Activate the Conda Environment in separate Windows Powershell or Ubuntu BASH Shell

03. Already INPUT DATA has been pre-created in "./pdf-policy-documents"

04. Please use the existing INPUT DATA as it is
    else re-create the INPUT DATA, Use "./run/run-data-maker.ps1" for Windows 11 with Powershell,
    or "./run/run-data-maker.sh" for Ubuntu 24 BASH Shell

05. Please use the FAISS-DB as it is, or to create a new embeddings or if FAISS-DB is not working, then use
    "./run/run-faiss-db-maker.ps1" for windows 11 powershell
    or "./run/run-faiss-db-maker.sh" for Ubuntu 24 BASH Shell

06. It will take from 30mins to  09.00 hours or slightly more to recreate the embeddings
=====================================================================================================

=====================================================================================================
# For Frontend SetUp
=====================================================================================================
01. Open a Separate Shell (Windows Powershell or Ubuntu BASH Shell) as per the case and usage
02. Please Navigate to the Frontend Context Root activate the conda nodejs environment and perform:
    npm install

=====================================================================================================

=====================================================================================================
# For Backend RUN or Execution
=====================================================================================================
01. Open a Separate Shell (Windows Powershell or Ubuntu BASH Shell) as per the case and usage
02. Please Navigate to the Backend Context Root and activate the shell
03. execute "./run/run-app.ps1" or execute the command
    uvicorn --host=localhost --app-dir=. app:app --port=8000

04. Please access the Backend API using MS-Edge/ Google Chrome/ Mozilla FireFox Browser with URL as:
    http://localhost:8000/home

=====================================================================================================

=====================================================================================================
# For Frontend RUN or Execution
=====================================================================================================
01. Open a Separate Shell (Windows Powershell or Ubuntu BASH Shell) as per the case and usage
02. Please Navigate to the Frontend Context Root and activate the shell
03. execute "./run/run-app.ps1" or execute the command
    execute npm start

04. Please access the Frontend GUI using MS-Edge/ Google Chrome/ Mozilla FireFox Browser with URL as:
    http://localhost:3000

=====================================================================================================


=====================================================================================================
# CONCLUSION
=====================================================================================================
01. Due to Time Constraints i have not enhanced the code, configuration, GUI parts
02. The code is all yours to play and please use it wisely and without any Pre-Judice
03. Also please forgive me if there is error or fault in this README file or Code 
=====================================================================================================

Thanks and Best Regards
Vinay Srinivasan
srinivasan_vinay@yahoo.com

