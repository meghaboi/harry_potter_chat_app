import streamlit as st
from google import genai
import io
from PIL import Image
from datetime import datetime
import requests

# Import custom modules
from image_generation import generate_image
from context_manager import chunk_text, get_active_chunk_context, search_context
from config import MODEL_ID, CHAT_MODEL

# Define Harry Potter characters
HARRY_POTTER_CHARACTERS = {
    "Harry Potter": "A brave, humble boy who survived Voldemort's attack. Known for his courage, loyalty, and occasional impulsiveness.",
    "Hermione Granger": "Exceptionally intelligent and studious witch, logical and detail-oriented. Values knowledge and preparation, fiercely loyal to her friends.",
    "Ron Weasley": "Loyal friend with a good sense of humor. Sometimes insecure but brave when it counts. From a large, loving wizard family.",
    "Albus Dumbledore": "Wise, enigmatic headmaster of Hogwarts. Speaks in riddles and believes in the power of love. Has deep knowledge of magic.",
    "Severus Snape": "Complex, stern Potions professor with a difficult past. Sharp-tongued and seemingly cold, but secretly protective.",
    "Rubeus Hagrid": "Half-giant gamekeeper with a big heart. Speaks in a distinct dialect, loves magical creatures, and is fiercely loyal to Dumbledore.",
    "Luna Lovegood": "Eccentric, dreamy student who believes in unusual creatures. Honest to a fault and unaffected by others' opinions.",
    "Draco Malfoy": "Arrogant Slytherin from a wealthy pure-blood family. Antagonistic but complex, struggles with the expectations placed on him.",
    "Minerva McGonagall": "Strict but fair Transfiguration professor and Head of Gryffindor. Proper, no-nonsense attitude but deeply cares for students.",
    "Sirius Black": "Harry's godfather, mischievous and rebellious. Intensely loyal, sometimes reckless, carries the trauma of his imprisonment in Azkaban."
}

# Define the character chat processing function with conversation context
def process_character_chat(prompt, message_placeholder, client):
    """
    Process a chat message with Gemini, handling context integration, character persona,
    and maintaining conversation history
    """
    try:
        # Determine what context to use from books
        book_context = ""
        
        if st.session_state.context_chunks:
            # Get context setting from tab3
            context_option = "Auto-search relevant chunks"  # Default
            if "context_option" in st.session_state:
                context_option = st.session_state.context_option
                
            if context_option == "Use active chunk only":
                book_context = get_active_chunk_context()
            elif context_option == "Auto-search relevant chunks":
                relevant_chunks = search_context(prompt)
                if relevant_chunks:
                    book_context = "\n\n".join(relevant_chunks)
                else:
                    # Fallback to active chunk if no relevant chunks found
                    book_context = get_active_chunk_context()
            else:  # Use all chunks
                book_context = "\n\n".join(st.session_state.context_chunks)
        else:
            book_context = st.session_state.context_text
        
        # Get character info
        character = st.session_state.selected_character
        character_description = HARRY_POTTER_CHARACTERS[character]
        
        # Check for any custom character settings
        custom_description = ""
        favorite_topics = ""
        speaking_style = "Neutral"
        
        if "character_custom_description" in st.session_state and st.session_state.character_custom_description:
            custom_description = st.session_state.character_custom_description
        
        if "favorite_topics" in st.session_state and st.session_state.favorite_topics:
            favorite_topics = st.session_state.favorite_topics
            
        if "speaking_style" in st.session_state:
            speaking_style = st.session_state.speaking_style
        
        # Build character prompt instructions
        character_instructions = f"""
        You are roleplaying as {character} from the Harry Potter series.
        
        Character description: {character_description}
        
        Your responses should authentically reflect this character's personality, knowledge, speech patterns, and worldview.
        You should respond as if you ARE this character, not as an AI pretending to be them.
        
        Don't use phrases like "As {character}, I would..." - just respond directly as the character would.
        
        When responding, consider:
        - The character's unique speech patterns and vocabulary
        - Their relationships with other characters
        - Their knowledge and experiences from the Harry Potter series
        - Their personality traits and values
        """
        
        if custom_description:
            character_instructions += f"\n\nAdditional character notes: {custom_description}"
            
        if favorite_topics:
            character_instructions += f"\n\nThis character particularly enjoys discussing: {favorite_topics}"
            
        character_instructions += f"\n\nThe character generally speaks in a {speaking_style.lower()} tone."
        
        # Build conversation history
        conversation_history = ""
        # Use character-specific history instead of messages
        if len(st.session_state[f"{character}_chat_history"]) > 0:  # Check if any chat history exists
            conversation_history = "Previous conversation history:\n"
            # Get the last 10 messages or all if fewer
            history_messages = st.session_state[f"{character}_chat_history"].copy()  # Make a copy
            if len(history_messages) > 10:
                history_messages = history_messages[-10:]  # Get last 10
                
            for msg in history_messages:
                role = "Human" if msg["role"] == "user" else character
                conversation_history += f"{role}: {msg['content']}\n"
        
        # Include context if available
        if book_context:
            final_prompt = f"""
            {character_instructions}
            
            Reference information from Harry Potter books:
            {book_context}
            
            {conversation_history}
            
            Remember to maintain continuity with the conversation history above.
            Now, please respond AS {character} to the following message:
            Human: {prompt}
            
            {character}:
            """
        else:
            final_prompt = f"""
            {character_instructions}
            
            {conversation_history}
            
            Remember to maintain continuity with the conversation history above.
            Now, please respond AS {character} to the following message:
            Human: {prompt}
            
            {character}:
            """
        
        # Save prompt to history
        st.session_state.prompt_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": f"Character Chat: {character}",
            "prompt": final_prompt
        })
        
        # Process with Gemini
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=final_prompt
        )
        response_text = response.text

        message_placeholder.markdown(response_text)
        
        # Store conversation in character-specific history only
        st.session_state[f"{character}_chat_history"].append({
            "role": "assistant",
            "content": response_text
        })

    except Exception as e:
        message_placeholder.error(f"Error: {str(e)}")

def initialize_gemini_client():
    """Initialize the Gemini client with the API key from session state"""
    if "api_key" in st.session_state and st.session_state.api_key:
        try:
            return genai.Client(api_key=st.session_state.api_key)
        except Exception as e:
            st.error(f"Error initializing Gemini client: {str(e)}")
            return None
    return None

# Set page configuration
st.set_page_config(page_title="Harry Potter Character Chat", layout="wide")

# Initialize session state variables
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "api_key_submitted" not in st.session_state:
    st.session_state.api_key_submitted = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "image_history" not in st.session_state:
    st.session_state.image_history = []

if "image_prompts" not in st.session_state:
    st.session_state.image_prompts = []

if "context_text" not in st.session_state:
    st.session_state.context_text = ""

if "context_chunks" not in st.session_state:
    st.session_state.context_chunks = []

if "active_chunk" not in st.session_state:
    st.session_state.active_chunk = 0

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 2000

if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

if "selected_character" not in st.session_state:
    st.session_state.selected_character = "Harry Potter"
    
# Initialize character-specific chat histories
for character in HARRY_POTTER_CHARACTERS:
    if f"{character}_chat_history" not in st.session_state:
        st.session_state[f"{character}_chat_history"] = []

# API Key Form Pop-up
if not st.session_state.api_key_submitted:
    st.markdown("## Welcome to Harry Potter Character Chat")
    
    # Create a form for API key input
    with st.form("api_key_form"):
        st.write("Please enter your Gemini API key to continue")
        api_key = st.text_input("Gemini API Key", type="password")
        
        # Submit button for the form
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            if api_key:
                st.session_state.api_key = api_key
                st.session_state.api_key_submitted = True
                st.experimental_rerun()
            else:
                st.error("Please enter a valid API key")
    
    # Add some guidance text
    st.markdown("""
    ### How to get a Gemini API key:
    1. Go to [Google AI Studio](https://aistudio.google.com/)
    2. Sign in with your Google account
    3. Navigate to API keys section
    4. Create a new API key
    
    This application uses Gemini to power Harry Potter character chats and image generation.
    """)
    
    # Stop the rest of the app from loading until API key is submitted
    st.stop()

# Initialize Gemini client with the API key
client = initialize_gemini_client()

if not client:
    st.error("Failed to initialize Gemini client. Please check your API key.")
    
    # Add a button to reset API key
    if st.button("Change API Key"):
        st.session_state.api_key_submitted = False
        st.session_state.api_key = ""
        st.experimental_rerun()
        
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Character Chat", "Image Generation", "Context Manager", "Prompt History", "Character Settings"])

# Tab 5: Character Settings
with tab5:
    st.header("Character Settings")
    
    st.markdown("""
    Select a character from the Harry Potter series to chat with. The character will respond based on their personality 
    and knowledge from the books.
    """)
    
    # Character selection
    selected_character = st.selectbox(
        "Choose a character to chat with:",
        list(HARRY_POTTER_CHARACTERS.keys()),
        index=list(HARRY_POTTER_CHARACTERS.keys()).index(st.session_state.selected_character)
    )
    
    # Update selected character
    if selected_character != st.session_state.selected_character:
        old_character = st.session_state.selected_character
        st.session_state.selected_character = selected_character
        
        st.experimental_rerun()
    
    # Show character description
    st.subheader(f"About {selected_character}")
    st.write(HARRY_POTTER_CHARACTERS[selected_character])
    
    # Advanced character settings
    with st.expander("Advanced Character Settings"):
        character_custom_description = st.text_area(
            "Custom character description (optional)",
            value=st.session_state.get("character_custom_description", ""),
            help="Add additional details about how you want the character to behave"
        )
        st.session_state.character_custom_description = character_custom_description
        
        favorite_topics = st.text_input(
            "Character's favorite topics (comma separated)",
            value=st.session_state.get("favorite_topics", ""),
            help="Topics this character would be particularly interested in discussing"
        )
        st.session_state.favorite_topics = favorite_topics
        
        speaking_style = st.select_slider(
            "Speaking formality",
            options=["Very Casual", "Casual", "Neutral", "Formal", "Very Formal"],
            value=st.session_state.get("speaking_style", "Neutral")
        )
        st.session_state.speaking_style = speaking_style
        
        # Clear character conversation history
        if st.button(f"Clear {selected_character}'s Conversation History"):
            st.session_state[f"{selected_character}_chat_history"] = []
            st.session_state.messages = []
            st.success(f"{selected_character}'s conversation history cleared!")
            st.experimental_rerun()
    
    # Add option to change API key
    with st.expander("API Settings"):
        st.write("Change your Gemini API key if needed")
        if st.button("Change API Key"):
            st.session_state.api_key_submitted = False
            st.session_state.api_key = ""
            st.success("API key reset. Please refresh the page.")
            st.experimental_rerun()

# Tab 4: Prompt History
with tab4:
    st.header("Prompt History")
    
    if st.session_state.prompt_history:
        st.write("This tab shows all prompts sent to AI models")
        
        for i, entry in enumerate(reversed(st.session_state.prompt_history)):
            with st.expander(f"{entry['timestamp']} - {entry['type']}"):
                st.code(entry['prompt'], language="text")
        
        if st.button("Clear Prompt History"):
            st.session_state.prompt_history = []
            st.success("Prompt history cleared!")
            st.experimental_rerun()
    else:
        st.info("No prompts have been sent yet. Interact with the application to see prompts here.")

# Tab 3: Context Manager
with tab3:
    st.header("Upload Harry Potter Books")
    
    st.markdown("""
    Upload Harry Potter book text files to provide context for the characters.
    The content will be split into manageable chunks for better processing.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        context_file = st.file_uploader("Upload a Harry Potter book", type=["txt"])
        chunk_size = st.slider("Chunk size (characters)", 500, 5000, st.session_state.chunk_size, 100)
        st.session_state.chunk_size = chunk_size
        
        if context_file is not None:
            text_content = context_file.read().decode("utf-8")
            st.session_state.context_text = text_content
            
            # Chunk the text
            chunks = chunk_text(text_content, chunk_size)
            st.session_state.context_chunks = chunks
            st.session_state.active_chunk = 0
            
            st.success(f"Harry Potter book uploaded and split into {len(chunks)} chunks!")
            
            if st.button("Clear Book Content"):
                st.session_state.context_text = ""
                st.session_state.context_chunks = []
                st.session_state.active_chunk = 0
                st.success("Book content cleared successfully!")
                st.experimental_rerun()
                
    with col2:
        if st.session_state.context_chunks:
            st.subheader("Book Content Overview")
            st.write(f"Total chunks: {len(st.session_state.context_chunks)}")
            st.write(f"Average chunk size: {sum(len(c) for c in st.session_state.context_chunks) // len(st.session_state.context_chunks)} characters")
            
            # Chunk navigation
            st.subheader("Browse Book Chunks")
            active_chunk = st.number_input("Current chunk", 1, len(st.session_state.context_chunks), st.session_state.active_chunk + 1)
            st.session_state.active_chunk = active_chunk - 1
            
            st.write("Active chunk preview:")
            st.text_area("Chunk content", get_active_chunk_context(), height=300, disabled=True)
            
            # Chat settings
            st.subheader("Chat Context Settings")
            context_option = st.radio(
                "How to use book content in chat:",
                ["Use active chunk only", "Auto-search relevant chunks", "Use all chunks"]
            )
            st.session_state.context_option = context_option
            
            if context_option == "Auto-search relevant chunks":
                st.info("The system will automatically find the most relevant book passages based on your query.")
            elif context_option == "Use all chunks":
                st.warning("Using all chunks may exceed context limits for very large books.")
                
        else:
            if st.session_state.context_text:
                st.info("Book content is loaded as a single chunk.")
                st.text_area("Book content", st.session_state.context_text, height=300, disabled=True)
            else:
                st.info("No Harry Potter book loaded. Upload a text file to provide context for the characters.")

# Tab 1: Chat with Character
with tab1:
    st.header(f"Chat with {st.session_state.selected_character}")
    
    # Character image and info
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Placeholder for character image (could be generated or uploaded)
        st.image("https://www.universalorlando.com/webdata/k2/en/us/files/Images/gds/uor-wwohp-logo-3-kids-clouds-key-art-hero-b.jpg")
    
    with col2:
        st.markdown(f"**{st.session_state.selected_character}**")
        st.markdown(HARRY_POTTER_CHARACTERS[st.session_state.selected_character])
        
        if st.session_state.context_chunks:
            st.info(f"Using {len(st.session_state.context_chunks)} book passages to inform {st.session_state.selected_character}'s responses.")
        elif st.session_state.context_text:
            st.info(f"Using book content to inform {st.session_state.selected_character}'s responses.")
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history for this character
    character_chat_history = st.session_state[f"{st.session_state.selected_character}_chat_history"] if f"{st.session_state.selected_character}_chat_history" in st.session_state else []
    
    for message in st.session_state[f"{st.session_state.selected_character}_chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Chat with {st.session_state.selected_character}..."):
        # Add user message to display
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add to session state
        if f"{st.session_state.selected_character}_chat_history" not in st.session_state:
            st.session_state[f"{st.session_state.selected_character}_chat_history"] = []
            
        st.session_state[f"{st.session_state.selected_character}_chat_history"].append({
            "role": "user",
            "content": prompt
        })

        # Process assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            process_character_chat(prompt, message_placeholder, client)

# Tab 2: Image Generation
with tab2:
    st.header("Generate Harry Potter Character Images")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Generate Character Image")
        character_for_image = st.selectbox("Select character to generate", list(HARRY_POTTER_CHARACTERS.keys()))
        
        image_prompt = st.text_area("Additional image details (optional)",
                                    height=100,
                                    key="new_image_prompt",
                                    placeholder="Example: standing in the Great Hall, holding a wand, with Hogwarts in the background")

        if st.button("Generate Character Image"):
            full_prompt = f"{character_for_image} from Harry Potter"
            if image_prompt:
                full_prompt += f", {image_prompt}"
                
            with st.spinner(f"Generating image of {character_for_image}..."):
                enhanced_prompt = f"Create a detailed sketch of {full_prompt}. Make it high quality and in the style of book illustrations."
                
                # Show the prompt being sent
                st.info(f"**Prompt sent to Gemini**: {enhanced_prompt}")

                image_data = generate_image(enhanced_prompt, client)

                if image_data:
                    st.session_state.current_image = image_data
                    st.session_state.image_history.append(image_data)
                    st.session_state.image_prompts.append(full_prompt)

                    st.success("Character image generated successfully!")
                else:
                    st.error("Failed to generate image.")

    with col2:
        st.subheader("Current Image")

        if st.session_state.current_image:
            st.text(f"Character: {st.session_state.current_image['prompt']}")
            st.text(f"Generated: {st.session_state.current_image['timestamp']}")

            st.image(st.session_state.current_image['image'], use_column_width=True)

            if st.session_state.current_image.get('text'):
                with st.expander("Image Description"):
                    st.markdown(st.session_state.current_image['text'])

            st.download_button(
                label="Download Image",
                data=st.session_state.current_image['image_data'],
                file_name=f"harry_potter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

            if st.button("Clear Current Image"):
                st.session_state.current_image = None
                st.experimental_rerun()
        else:
            st.info("No character image generated yet.")

        st.subheader("Image History")
        if st.session_state.image_history:
            for i, img in enumerate(reversed(st.session_state.image_history[-5:])):
                with st.expander(f"Image {len(st.session_state.image_history) - i}"):
                    st.text(f"Character: {img['prompt']}")
                    st.text(f"Generated: {img['timestamp']}")
                    st.image(img['image'], use_column_width=True)
        else:
            st.info("No image history yet.")

st.markdown("---")
st.markdown("Harry Potter Character Chat by Meghanadh Pamidi")