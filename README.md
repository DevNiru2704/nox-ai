
# NOX: Your Advanced AI Companion

NOX is an advanced AI-powered virtual companion that combines real-time face tracking, natural language processing, 3D animation, speech synthesis, and generative AI capabilities to provide a unique and personalized interactive experience.

## About NOX

NOX is designed to be your intelligent, adaptable, and engaging digital companion. With its advanced capabilities, NOX can understand your emotions, engage in meaningful conversations, and assist you with various tasks. Whether you're looking for a friendly chat, need help with problem-solving, or want to explore creative endeavors, NOX is here to enhance your digital experience.

## Features

- **Real-time Face Tracking & Expression Analysis**: NOX analyzes your expressions via webcam to understand your mood and emotions, allowing for more empathetic interactions.
- **Natural Language Processing & Chat Interaction**: Engage in deep, context-aware conversations with NOX on a wide range of topics.
- **3D Animation & Character Modeling**: Interact with NOX's customizable 3D avatar, complete with realistic animations and expressions.
- **Speech Synthesis & Voice Modulation**: Enjoy lifelike conversations with NOX's customizable voice, making each interaction feel natural and personal.
- **Generative AI**: Ask NOX to create images, videos, and files, expanding the possibilities of your interactions.

## Technologies Used

- Face Tracking: MediaPipe Face Mesh, OpenFace 2.0
- Emotion Recognition: DeepFace, FER
- NLP: GPT-4 / GPT-NeoX, LangChain, Rasa
- 3D Animation: Avatarify, NVIDIA Omniverse Audio2Face, Blender
- Speech Synthesis: Coqui TTS, Microsoft Azure Speech, Descript Overdub
- Generative AI: Stable Diffusion, DALL-E, Runway Gen-2, First Order Motion Model
- Backend: Streamlit, FastAPI

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nox-ai-companion.git
   cd nox-ai-companion
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   AZURE_SPEECH_KEY=your_azure_speech_key
   ```

4. Run the application:
   ```
   streamlit run nox_app.py
   ```

## Usage

1. Open your web browser and navigate to `http://localhost:8501`.
2. Grant permission for the application to access your webcam when prompted.
3. Start interacting with NOX through text or voice input.
4. Explore NOX's various capabilities, from engaging in conversations to generating creative content.

## Customizing NOX

NOX is designed to be adaptable to your preferences. You can customize various aspects of NOX's personality, appearance, and capabilities:

- Adjust NOX's voice and speech patterns in the `speech_synthesis.py` file.
- Modify NOX's 3D avatar appearance and animations in the `avatar_customization.py` file.
- Fine-tune NOX's conversation style and knowledge base in the `nox_personality.py` file.

speech_synthesis.py
```
import pyttsx3
import speech_recognition as sr

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    return recognizer.recognize_google(audio)
```

text_to_image.py
```
import openai
from PIL import Image
import requests
from io import BytesIO

openai.api_key = 'YOUR_OPENAI_API_KEY'

def text_to_image(prompt):
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    image_url = response['data'][0]['url']
    image_data = requests.get(image_url).content
    image = Image.open(BytesIO(image_data))
    image.show()

text_to_image("a cozy cabin in the snowy mountains at sunset")
```
avatar_customization.py
```
import avatarify
import numpy as np

class AvatarCustomizer:
    def __init__(self):
        self.avatar_model = avatarify.load_avatar_model()

    def set_avatar_appearance(self, model_file):
        self.avatar_model = avatarify.load_avatar_model(model_file)

    def apply_expression(self, expression_type, intensity=1.0):
        expression = np.array([expression_type, intensity])
        avatarify.apply_expression(self.avatar_model, expression)

    def animate_avatar(self, animation_data):
        avatarify.animate(self.avatar_model, animation_data)

    def reset_avatar(self):
        self.avatar_model = avatarify.load_avatar_model()

# Example usage
if __name__ == "__main__":
    avatar_customizer = AvatarCustomizer()
    avatar_customizer.set_avatar_appearance("custom_avatar_model.obj")
    avatar_customizer.apply_expression("smile", intensity=0.8)
    animation_sequence = {"type": "wave", "duration": 2.0}
    avatar_customizer.animate_avatar(animation_sequence)
```

nox_personality.py
```
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class NoxPersonality:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.personality_traits = {}

    def load_personality_traits(self, traits_file):
        with open(traits_file, 'r') as file:
            self.personality_traits = json.load(file)

    def generate_response(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors='pt')
        response_ids = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response

    def customize_conversation_style(self, style):
        self.personality_traits['conversation_style'] = style

    def get_personality_summary(self):
        return self.personality_traits

# Example usage
if __name__ == "__main__":
    nox = NoxPersonality()
    nox.load_personality_traits("personality_traits.json")
    user_input = "What's your favorite hobby?"
    response = nox.generate_response(user_input)
    print(response)
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe)
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
- [DeepFace](https://github.com/serengil/deepface)
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- [LangChain](https://github.com/hwchase17/langchain)
- [Avatarify](https://github.com/alievk/avatarify)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
