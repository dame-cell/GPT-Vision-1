# GPT-Vision-1

We all like Moondream the 1 billion Parameters Vision Language model that kicks ass 
Well how about something smaller a 200 Million Parameter Vision Language model which is not as good as I would like it to be 

# Model architecture 

This Model follows the same architecture as LLava 

# Training Details 

- We first pre-train the model while freezing the LLM and the Vision Transformers and only pre-training the projector which is a simple MLP nothing unqiue 
- Then save the pre-train model to huggingface 
- Load the pre-train model for fine-tuning but this time we froze only the Vision Transformers  




