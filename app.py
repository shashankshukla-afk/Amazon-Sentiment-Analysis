import streamlit as st
import torch
import plotly.graph_objects as go
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# 1. LOAD MODEL

model_path = "./distilbert_sentiment_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)
model.eval()
# 2. SESSION STATE


if "history" not in st.session_state:
    st.session_state.history = []

if "sentiment" not in st.session_state:
    st.session_state.sentiment = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

# 3. LABEL MAP

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
# 4. PREDICTION FUNCTION
def predict_sentiment(review):
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]
    predicted_class = torch.argmax(
        probabilities
    ).item()
    sentiment = label_map[predicted_class]
    confidence = probabilities[
        predicted_class
    ].item()
    return sentiment, confidence

# 5. SENTIMENT GAUGE

def create_sentiment_gauge(sentiment):
    # Needle position


    if sentiment == "Negative":
        angle = 150
    elif sentiment == "Neutral":
        angle = 90
    else:
        angle = 30
    # Convert angle to radians
    angle_rad = math.radians(angle)
    radius = 1
    needle_length = 0.78

    # Needle endpoint
    needle_x = (
        needle_length * math.cos(angle_rad)
    )
    needle_y = (
        needle_length * math.sin(angle_rad)
    )
    # Create figure
    fig = go.Figure()
    # Gauge colored zones
    zones = [
        # Negative
        (180, 120, "#ff4b4b"),
        # Neutral
        (120, 60, "#ffd43b"),
        # Positive
        (60, 0, "#21c354")
    ]
    for start, end, color in zones:
        angles = [
            math.radians(a)
            for a in range(end, start + 1)
        ]
        x_values = [
            radius * math.cos(a)
            for a in angles
        ]
        y_values = [
            radius * math.sin(a)
            for a in angles
        ]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(
                    color=color,
                    width=35
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )
    # Needle
    fig.add_trace(
        go.Scatter(
            x=[
                0,
                needle_x
            ],
            y=[
                0,
                needle_y
            ],
            mode="lines",
            line=dict(
                color="black",
                width=7
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )
    # Needle center

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(
                color="black",
                size=18
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )
    # Negative label
    fig.add_annotation (
        x=-0.95,
        y=-0.08,
        text="<b>Negative</b>",
        showarrow=False,
        font=dict(
            size=16
        )
    )
    # Neutral label
    fig.add_annotation(
        x=0,
        y=1.18,
        text="<b>Neutral</b>",
        showarrow=False,
        font=dict(
            size=16
        )
    )
    # Positive label

    fig.add_annotation(
        x=0.95,
        y=-0.08,
        text="<b>Positive</b>",
        showarrow=False,
        font=dict(
            size=16
        )
    )
    # Current sentiment

    fig.add_annotation(
        x=0,
        y=-0.32,
        text=f"<b>{sentiment}</b>",
        showarrow=False,
        font=dict(
            size=22
        )
    )
    # Gauge layout
    fig.update_layout(
        height=400,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),
        xaxis=dict(
            range=[-1.2, 1.2],
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-0.45, 1.3],
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# 6. SIDEBAR NAVIGATION

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "📜 History",
        "Guide"
    ]
)
# 7. HOME PAGE

if page == "🏠 Home":
    st.title("Amazon Review Sentiment Analyzer")
    st.write(
        "Enter a customer review and let DistilBERT "
        "predict its sentiment."
    )
    # Review input

    review = st.text_area(
        "Enter a customer review:",
        placeholder="Type your review here...",
        height=150
    )
    # Analyze button

    if st.button(
        "Analyze Sentiment",
        use_container_width=True
    ):
        if review.strip():
            # Get prediction from DistilBERT
            sentiment, confidence = predict_sentiment(
                review
            )
            # Store prediction
            st.session_state.sentiment = sentiment
            st.session_state.confidence = confidence
            # Save to history

            st.session_state.history.append({
                "Review": review,
                "Sentiment": sentiment,
                "Confidence": confidence

            })
        else:
            st.warning(
                "Please enter a review before analyzing."
            )
    # SHOW RESULT

    if st.session_state.sentiment is not None:
        st.subheader("Analysis Result")

        # Sentiment message
        if st.session_state.sentiment == "Positive":
            st.success(
                "😊 Positive Sentiment"
            )
        elif st.session_state.sentiment == "Negative":
            st.error(
                "😞 Negative Sentiment"
            )
        else:
            st.warning(
                "😐 Neutral Sentiment"
            )
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Sentiment",
                st.session_state.sentiment
            )
        with col2:
            st.metric(
                "Confidence",
                f"{st.session_state.confidence * 100:.2f}%"
            )
        # SENTIMENT GAUGE
        st.subheader("Sentiment Gauge")
        gauge = create_sentiment_gauge(
            st.session_state.sentiment
        )
        st.plotly_chart(
            gauge,
            use_container_width=True
        )


elif page == "📜 History":
    st.title("📜 Analysis History")
    if st.session_state.history:
        st.write(
            f"Total analyses: "
            f"{len(st.session_state.history)}"
        )
        # Display history
        for i, item in enumerate(
            reversed(st.session_state.history),
            start=1
        ):
            st.subheader(
                f"Analysis {i}"
            )
            st.write(
                "**Review:**",
                item["Review"]
            )
            st.write(
                "**Sentiment:**",
                item["Sentiment"]
            )
            st.write(
                "**Confidence:**",
                f"{item['Confidence'] * 100:.2f}%"
            )
            st.divider()
        # Clear history
        if st.button(
            "🗑️ Clear History",
            use_container_width=True
        ):
            st.session_state.history = []
            st.session_state.sentiment = None
            st.session_state.confidence = None
            st.rerun()
    else:
        st.info(
            "No analysis history yet."
        )

elif page == "Guide":

    st.title("📖 How to Use This Web App")
    st.write(
        "This web application analyzes customer reviews and "
        "predicts whether the sentiment is Negative, Neutral, or Positive."
    )
    st.header("1️⃣ Enter Your Review")
    st.write(
        "Go to the **🏠 Home** page and enter or paste your "
        "customer review into the text box."
    )
    st.code(
        "Example: The product quality is excellent "
        "and I am very happy with my purchase.",
        language="text"
    )
    st.header("2️⃣ Analyze the Review")
    st.write(
        "After entering your review, click the "
        "**Analyze Sentiment** button."
    )
    st.write(
        "The AI model will process your review and predict "
        "its sentiment."
    )
    st.header("3️⃣ Understand the Result")
    st.write(
        "The application can classify your review into three categories:"
    )
    st.markdown(
        """
        🔴 **Negative**  
        The review expresses a negative opinion.

        🟡 **Neutral**  
        The review expresses a neutral or mixed opinion.

        🟢 **Positive**  
        The review expresses a positive opinion.
        """
    )
    st.header("4️⃣ Understand the Sentiment Gauge")

    st.write(
        "The sentiment gauge visually shows the predicted sentiment."
    )

    st.markdown(
        """
        🔴 **Left** → Negative

        🟡 **Center** → Neutral

        🟢 **Right** → Positive
        """
    )
    st.write(
        "The needle automatically moves to the corresponding "
        "sentiment after the review is analyzed."
    )
    st.header("5️⃣ Check the Confidence")

    st.write(
        "The confidence value shows how confident the AI model "
        "is in its prediction."
    )

    st.info(
        "A higher confidence does not guarantee that the prediction "
        "is correct. The model can occasionally make incorrect predictions."
    )

    st.header("🚀 Quick Guide")

    st.code(
        """
1. Open Home
2. Enter a customer review
3. Click Analyze Sentiment
4. View the predicted sentiment
5. Check the gauge
6. Check the confidence
        """
    )
