# AI-Driven Streamlit App

## Overview

Developed an AI pipeline that processes images to segment objects, identify them in real-world contexts, extract text, and generate summaries using advanced natural language processing models. The project also includes a web application built with Streamlit where users can upload images to:

- **Segment objects** using ResNet-50.
- **Identify real-world objects**.
- **Extract text** using Optical Character Recognition (OCR).
- **Generate automated summaries** using GPT-2/GPT-Neo.

## Features

- **Interactive UI**: A responsive and user-friendly interface built with Streamlit.
- **Data Visualization**: Interactive charts and plots to visualize data in real-time.
- **Machine Learning Integration**: Integration with machine learning models for data predictions and insights.
- **Media Upload**: Functionality to upload and manage media files, including images and videos.

## Technologies Used

- **PyTorch**
- **Streamlit**
- **ResNet-50** (for real-world object identification)
- **OCR** (Optical Character Recognition)
- **GPT-2/GPT-Neo** (for text generation and summarization)

## Installation

To set up and run the `AI-Driven-Streamlit-App` on your local machine, follow these steps:

### Prerequisites

Ensure you have Python 3.7 or higher installed. Check your Python version with:

```bash
python --version
```

### Clone the Repository

Clone the repository from GitHub:

```bash
git clone https://github.com/MithunKumar09/AI-Driven-Image-Processing-Pipeline-with-Real-World-Object-Identification-And-Automated-Text-Generate.git
cd AiInternTask
```

### Set Up a Virtual Environment

Create and activate a virtual environment:

On Windows:

```bash
python -m venv newenv2
newenv2\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv newenv2
source newenv2/bin/activate
```

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Create a `.env` File

Create a `.env` file in the root directory to store sensitive configuration details. Ensure it includes necessary environment variables such as API keys or database credentials. The `.env` file should look something like this:

```dotenv
# Example environment variables
API_KEY=your_api_key_here
DATABASE_URL=your_database_url_here
SECRET_KEY=your_secret_key_here
```

### Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

## Project Structure

```plaintext
├── app.py                    # The main Streamlit application file that initializes and runs the app.
├── requirements.txt          # A list of Python packages required for the project.
├── media/                    # Directory for storing uploaded media files.
├── .gitignore                # Specifies files and directories to be ignored by Git.
├── .env                      # Environment variables file (not included in the repository for security reasons).
└── models/                   # Contains the machine learning models used in the application.
    ├── segmentation_model.py
    ├── identification_model.py
    ├── text_extraction_model.py
    └── summarization_model.py
```

## Usage

- **Access the App**: Open your web browser and go to [http://localhost:8501](http://localhost:8501) to interact with the Streamlit app.
- **Upload Media**: Use the media upload functionality to add images and videos.
- **View Data Visualizations**: Explore dynamic charts and plots to analyze data.
- **Interact with Machine Learning Models**: Input data to receive predictions and insights from the integrated models.

## Contributing

Contributions to the AI-Driven-Streamlit-App are welcome. To contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or fix:

   ```bash
   git checkout -b feature/your-feature
   ```

3. Make your changes and commit them:

   ```bash
   git commit -am 'Add new feature'
   ```

4. Push your changes to your forked repository:

   ```bash
   git push origin feature/your-feature
   ```

5. Create a Pull Request to merge your changes into the main repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or further information, please contact:

- **Email**: [mithunkumaar098@gmail.com](mailto:mithunkumaar098@gmail.com)
- **GitHub**: [MithunKumar09](https://github.com/MithunKumar09)

## Acknowledgements

- **Streamlit**: For providing a robust framework for building interactive applications.
- **Open Source Libraries**: Various Python libraries and tools used throughout this project. 
