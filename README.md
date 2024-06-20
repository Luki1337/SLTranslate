# SLTranslate

SLTranslate is a web application that translates sign language into English letters from images. It uses an adapted version of the pytorch ConvNext CNN machine learning model (https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html#torchvision.models.convnext_tiny) to analyze uploaded images and provide corresponding translations.

## Frontend

The frontend of SLTranslate is built using JavaScript (React) and includes the following steps to set up and run:

1. **Clone the Repository:**

   ```
   git clone https://github.com/Luki1337/SLTranslate
   cd SLTranslate
   ```

1. **Install the dependencies**
    ```
    npm install
    ```
    This will install all necessary dependencies listed in package.json.

    Note: Node.js needs to be installed for the npm package manager to work. You can download Node.js from nodejs.org.

4. **Running the Frontend**
    To start the frontend server:
    ```
    npm start
    ```
    The frontend will be accessible at http://localhost:3000.

## Backend

The backend of SLTranslate is built using Python (Flask) and requires a specific conda environment to run the server:

1. **Create and Activate Conda Environment**
   
    ```
    conda env create -f environment.yml
    conda activate SignTranslate
    ```
    
    This will create a conda environment named SignTranslate with all required dependencies.

    Note: Anaconda or Miniconda needs to be installed. You can download Anaconda from anaconda.com or Miniconda from docs.conda.io/en/latest/miniconda.html.

3. **Run the backend**

    To start the backend server:
   
    ```
    python server.py
    ```

    The backend will then be accessible at http://localhost:5000.

## Additional Notes 

Ensure that ports 3000 for the frontend and 5000 for the backend are free.

