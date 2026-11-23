
import streamlit as st
import cv2
import tempfile
import os
import av
import pandas as pd

from ultralytics import YOLO
from tracker import ObjectTracker

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode
)


# =================================
# PAGE CONFIGURATION
# =================================

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)


# =================================
# PASTEL BLUE THEME
# =================================

st.markdown("""
<style>

.stApp {
    background-color: #F0F7FF;
}

section[data-testid="stMain"] {
    background-color: #F0F7FF;
}

section[data-testid="stSidebar"] {
    background-color: #DCEBFA;
}

h1, h2, h3 {
    color: #294C70 !important;
}

.stApp p, .stApp label {
    color: #405D78;
}

.stButton > button {
    background-color: #A9CFF5;
    color: #183B5B;
    border: 1px solid #8DBBEA;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #8DBBEA;
    color: #183B5B;
    border: 1px solid #6FA7DF;
}

[data-testid="stMetric"] {
    background-color: #DCEBFA;
    border: 1px solid #C3DDF5;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stFileUploader"] {
    background-color: #E5F1FD;
    border: 1px dashed #8DBBEA;
    border-radius: 12px;
    padding: 12px;
}

.stSelectbox [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #8DBBEA !important;
    border-radius: 8px !important;
}

.stSelectbox [data-baseweb="select"] span {
    color: #000000 !important;
}

.stSelectbox [data-baseweb="select"] svg {
    fill: #000000 !important;
}

[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="menu"] li {
    color: #000000 !important;
}

[data-baseweb="menu"] li:hover {
    background-color: #DCEBFA !important;
}

.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =================================
# LOAD YOLO MODEL
# =================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# =================================
# OBJECT COUNTING FUNCTION
# =================================

def calculate_object_counts(detection_history):

    total_unique_objects = len(detection_history)

    people_detected = 0
    object_counts = {}

    for obj in detection_history.values():

        label = obj.get(
            "Object Name",
            obj.get("Object", "")
        )

        if label.lower() == "person":

            people_detected += 1

        else:

            object_counts[label] = (
                object_counts.get(label, 0) + 1
            )

    total_objects_detected = sum(
        object_counts.values()
    )

    return {
        "total_unique_objects": total_unique_objects,
        "people_detected": people_detected,
        "total_objects_detected": total_objects_detected,
        "object_counts": object_counts
    }


# =================================
# CROWD MONITORING FUNCTION
# =================================

def calculate_crowd_status(current_people, crowd_limit):

    percentage = (
        current_people / crowd_limit
    ) * 100

    if percentage <= 50:

        status = "Low"

    elif percentage <= 75:

        status = "Moderate"

    elif percentage <= 100:

        status = "High"

    else:

        status = "Overcrowded"

    return {
        "crowd_status": status,
        "crowd_percentage": round(percentage, 2)
    }


# =================================
# WEBCAM PROCESSOR
# =================================

class WebcamProcessor(VideoProcessorBase):

    def __init__(self):

        self.tracker = ObjectTracker(model)

        self.total_ids = set()

        self.total_frames = 0

        self.current_count = 0

        self.current_people_count = 0

        self.detection_history = {}

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        processed_frame, tracked_objects = (
            self.tracker.track_objects(img)
        )

        # =================================
        # CURRENT OBJECT COUNT
        # =================================

        self.current_count = len(
            tracked_objects
        )

        # =================================
        # CURRENT PEOPLE COUNT
        # =================================

        self.current_people_count = sum(
            1 for obj in tracked_objects
            if obj["label"].lower() == "person"
        )

        # =================================
        # FRAME COUNTING
        # =================================

        self.total_frames += 1

        # =================================
        # DETECTION HISTORY
        # =================================

        for obj in tracked_objects:

            object_id = obj["id"]

            self.total_ids.add(object_id)

            self.detection_history[object_id] = {

                "Tracking ID": object_id,

                "Object": obj["label"],

                "Confidence": round(
                    obj["confidence"],
                    2
                )
            }

        total_count = len(
            self.total_ids
        )

        # =================================
        # VIDEO STATISTICS BACKGROUND
        # =================================

        cv2.rectangle(
            processed_frame,
            (10, 10),
            (390, 145),
            (30, 30, 30),
            -1
        )

        # Currently tracked objects

        cv2.putText(
            processed_frame,
            f"Currently Tracked: {self.current_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # Total unique objects

        cv2.putText(
            processed_frame,
            f"Total Unique Objects: {total_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # Frames processed

        cv2.putText(
            processed_frame,
            f"Frames Processed: {self.total_frames}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            processed_frame,
            format="bgr24"
        )


# =================================
# SESSION STATE
# =================================

if "input_type" not in st.session_state:

    st.session_state.input_type = None


# =================================
# FRONT PAGE
# =================================

if st.session_state.input_type is None:

    st.markdown(
        """
<div style="text-align:center; padding:35px 10px;">
<h1 style="color:#24476B; font-size:38px;">
🎯 Object Detection and Tracking System
</h1>
<p style="color:#52677D; font-size:17px;">
Detect, track and analyze objects using YOLOv8 and ByteTrack.
</p>
</div>
""",
        unsafe_allow_html=True
    )

    st.subheader("Select Video Input")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📁 Upload Video")

        st.write(
            "Upload a video to detect and track objects."
        )

        if st.button(
            "Choose Upload Video",
            use_container_width=True
        ):

            st.session_state.input_type = "Upload Video"

            st.rerun()

    with col2:

        st.markdown("### 📷 Webcam")

        st.write(
            "Use your webcam for real-time object detection."
        )

        if st.button(
            "Start Webcam",
            use_container_width=True
        ):

            st.session_state.input_type = "Webcam"

            st.rerun()


# =================================
# UPLOAD VIDEO
# =================================

elif st.session_state.input_type == "Upload Video":

    if st.button("⬅ Back to Home"):

        st.session_state.input_type = None

        st.rerun()

    st.subheader("📁 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:

        st.success(
            f"Video selected: {uploaded_file.name}"
        )

        if st.button("▶ Start / Replay Detection"):

            video_tracker = ObjectTracker(model)

            extension = os.path.splitext(
                uploaded_file.name
            )[1]

            if not extension:

                extension = ".mp4"

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            )

            temp_path = temp_file.name

            cap = None

            try:

                # Save uploaded video

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_file.close()

                # Open video

                cap = cv2.VideoCapture(
                    temp_path
                )

                if not cap.isOpened():

                    st.error(
                        "Unable to open the uploaded video."
                    )

                else:

                    video_display = st.empty()

                    st.subheader(
                        "📊 Live Detection Statistics"
                    )

                    col1, col2, col3 = st.columns(3)

                    current_metric = col1.empty()

                    unique_metric = col2.empty()

                    frame_metric = col3.empty()

                    # =================================
                    # INITIALIZE COUNTERS
                    # =================================

                    total_ids = set()

                    total_frames = 0

                    detection_history = {}

                    final_current_count = 0

                    final_people_count = 0

                    # =================================
                    # PROCESS VIDEO FRAMES
                    # =================================

                    while cap.isOpened():

                        success, frame = cap.read()

                        if not success:

                            break

                        processed_frame, tracked_objects = (
                            video_tracker.track_objects(frame)
                        )

                        total_frames += 1

                        # =================================
                        # CURRENT COUNTS
                        # =================================

                        final_current_count = len(
                            tracked_objects
                        )

                        final_people_count = sum(
                            1 for obj in tracked_objects
                            if obj["label"].lower() == "person"
                        )

                        # =================================
                        # DETECTION HISTORY
                        # =================================

                        for obj in tracked_objects:

                            track_id = obj["id"]

                            total_ids.add(track_id)

                            detection_history[track_id] = {

                                "Tracking ID": track_id,

                                "Object Name": obj["label"],

                                "Confidence": (
                                    f'{obj["confidence"] * 100:.2f}%'
                                )
                            }

                        # =================================
                        # DISPLAY VIDEO
                        # =================================

                        frame_rgb = cv2.cvtColor(
                            processed_frame,
                            cv2.COLOR_BGR2RGB
                        )

                        video_display.image(
                            frame_rgb,
                            channels="RGB",
                            use_container_width=True
                        )

                        # =================================
                        # LIVE STATISTICS
                        # =================================

                        current_metric.metric(
                            "Currently Tracked",
                            len(tracked_objects)
                        )

                        unique_metric.metric(
                            "Total Unique Objects",
                            len(total_ids)
                        )

                        frame_metric.metric(
                            "Frames Processed",
                            total_frames
                        )

                    st.success(
                        "Video processing completed!"
                    )

                    # =================================
                    # OBJECT & PEOPLE COUNTING
                    # =================================

                    st.divider()

                    st.subheader(
                        "📊 Object & People Counting"
                    )

                    count_results = calculate_object_counts(
                        detection_history
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Unique Objects",
                        count_results["total_unique_objects"]
                    )

                    col2.metric(
                        "People Detected",
                        count_results["people_detected"]
                    )

                    col3.metric(
                        "Total Objects Detected",
                        count_results["total_objects_detected"]
                    )

                    # =================================
                    # INDIVIDUAL OBJECT COUNTS
                    # =================================

                    st.markdown(
                        "### Individual Object Counts"
                    )

                    if count_results["object_counts"]:

                        object_count_df = pd.DataFrame(
                            [
                                {
                                    "Object": label,
                                    "Count": count
                                }

                                for label, count
                                in count_results["object_counts"].items()
                            ]
                        )

                        st.dataframe(
                            object_count_df,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.info(
                            "No other objects detected."
                        )

                    # =================================
                    # DETECTION SUMMARY
                    # =================================

                    st.divider()

                    st.subheader(
                        "📋 Detection Summary"
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Unique Objects",
                        len(total_ids)
                    )

                    col2.metric(
                        "Total Frames Processed",
                        total_frames
                    )

                    col3.metric(
                        "Final Currently Tracked",
                        final_current_count
                    )

                    # =================================
                    # DETECTION HISTORY
                    # =================================

                    st.divider()

                    st.subheader(
                        "📋 Object Detection History"
                    )

                    if detection_history:

                        history_df = pd.DataFrame(
                            list(detection_history.values())
                        )

                        st.dataframe(
                            history_df,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        history_df = pd.DataFrame(
                            columns=[
                                "Tracking ID",
                                "Object Name",
                                "Confidence"
                            ]
                        )

                        st.info(
                            "No objects were detected."
                        )

                    # =================================
                    # DOWNLOAD REPORTS
                    # =================================

                    st.divider()

                    st.subheader(
                        "📥 Download Detection Reports"
                    )

                    history_csv = history_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    summary_df = pd.DataFrame([{

                        "Total Unique Objects": len(total_ids),

                        "Total Frames Processed": total_frames,

                        "Final Currently Tracked": final_current_count,

                        "People Detected":
                            count_results["people_detected"],

                        "People Currently Visible":
                            final_people_count,

                        "Total Objects Detected":
                            count_results["total_objects_detected"]

                    }])

                    summary_csv = summary_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.download_button(
                            label="📄 Download Detection History",
                            data=history_csv,
                            file_name="detection_history.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col2:

                        st.download_button(
                            label="📊 Download Detection Summary",
                            data=summary_csv,
                            file_name="detection_summary.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            except Exception as e:

                st.error(
                    f"Video processing error: {e}"
                )

            finally:

                if cap is not None:

                    cap.release()

                if not temp_file.closed:

                    temp_file.close()

                if os.path.exists(temp_path):

                    os.unlink(temp_path)


# =================================
# WEBCAM
# =================================

elif st.session_state.input_type == "Webcam":

    # =================================
    # INITIALIZE WEBCAM SESSION
    # =================================

    if "webcam_session_id" not in st.session_state:

        st.session_state.webcam_session_id = 0

    if "webcam_report" not in st.session_state:

        st.session_state.webcam_report = None

    # =================================
    # NAVIGATION BUTTONS
    # =================================

    if st.button("⬅ Back to Home"):

        st.session_state.input_type = None

        st.session_state.webcam_session_id += 1

        st.session_state.webcam_report = None

        st.rerun()

    if st.button("🔄 Reset Webcam Detection"):

        st.session_state.webcam_session_id += 1

        st.session_state.webcam_report = None

        st.rerun()

    # =================================
    # WEBCAM INTERFACE
    # =================================

    st.subheader(
        "📷 Live Object Detection"
    )

    st.write(
        "Start your webcam to detect and track objects in real time. "
        "Stop the webcam to view your final detection report."
    )

    # =================================
    # CROWD CAPACITY SETTINGS
    # =================================

    crowd_limit = st.number_input(
        "Maximum Crowd Capacity",
        min_value=1,
        value=10,
        step=1
    )

    # =================================
    # WEBRTC STREAMER
    # =================================

    webcam_col, empty_col = st.columns([110, 15])

    with webcam_col:

     ctx = webrtc_streamer(

        key=f"object-tracking-{st.session_state.webcam_session_id}",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=WebcamProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )
     
    st.info(
        "Your webcam feed displays detected objects, "
        "bounding boxes, tracking IDs and live statistics."
    )

    # =================================
    # LIVE STATISTICS AND FINAL REPORT
    # =================================

    @st.fragment(run_every="1s")
    def show_webcam_results():

        processor = ctx.video_processor

        # =================================
        # SAVE LATEST WEBCAM STATISTICS
        # =================================

        if processor is not None:

            count_results = calculate_object_counts(
                processor.detection_history
            )

            crowd_results = calculate_crowd_status(
                processor.current_people_count,
                crowd_limit
            )

            st.session_state.webcam_report = {

                "total_unique_objects":
                    count_results["total_unique_objects"],

                "people_detected":
                    count_results["people_detected"],

                "total_objects_detected":
                    count_results["total_objects_detected"],

                "object_counts":
                    count_results["object_counts"],

                "total_frames":
                    processor.total_frames,

                "currently_tracked":
                    processor.current_count,

                "people_currently_visible":
                    processor.current_people_count,

                "crowd_status":
                    crowd_results["crowd_status"],

                "crowd_percentage":
                    crowd_results["crowd_percentage"],

                "crowd_limit":
                    crowd_limit,

                "detection_history":
                    list(
                        processor.detection_history.values()
                    )
            }

        report = st.session_state.webcam_report

        # =================================
        # LIVE DETECTION STATISTICS
        # =================================

        if ctx.state.playing:

            st.divider()

            st.subheader(
                "📊 Live Detection Statistics"
            )

            if report is not None:

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Currently Tracked",
                    report["currently_tracked"]
                )

                col2.metric(
                    "Total Unique Objects",
                    report["total_unique_objects"]
                )

                col3.metric(
                    "Frames Processed",
                    report["total_frames"]
                )

                # =================================
                # LIVE OBJECT & PEOPLE COUNTING
                # =================================

                st.subheader(
                    "📊 Object & People Counting"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Unique Objects",
                    report["total_unique_objects"]
                )

                col2.metric(
                    "People Detected",
                    report["people_detected"]
                )

                col3.metric(
                    "Total Objects Detected",
                    report["total_objects_detected"]
                )

                # =================================
                # PEOPLE CURRENTLY VISIBLE
                # =================================

                st.metric(
                    "People Currently Visible",
                    report["people_currently_visible"]
                )

                # =================================
                # CROWD MONITORING
                # =================================

                st.subheader(
                    "👥 Crowd Monitoring"
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Crowd Status",
                    report["crowd_status"]
                )

                col2.metric(
                    "Crowd Capacity Used",
                    f'{report["crowd_percentage"]}%'
                )

                st.caption(
                    f'Maximum Crowd Capacity: {report["crowd_limit"]}'
                )

                # =================================
                # INDIVIDUAL OBJECT COUNTS
                # =================================

                st.markdown(
                    "### Individual Object Counts"
                )

                if report["object_counts"]:

                    object_count_df = pd.DataFrame(
                        [
                            {
                                "Object": label,
                                "Count": count
                            }

                            for label, count
                            in report["object_counts"].items()
                        ]
                    )

                    st.dataframe(
                        object_count_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No other objects detected."
                    )

            else:

                st.info(
                    "Waiting for webcam frames..."
                )

        # =================================
        # FINAL REPORT AFTER STOPPING
        # =================================

        elif report is not None and report["total_frames"] > 0:

            st.success(
                "✅ Webcam recording completed successfully!"
            )

            # =================================
            # DETECTION SUMMARY
            # =================================

            st.divider()

            st.subheader(
                "📋 Detection Summary"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Unique Objects",
                report["total_unique_objects"]
            )

            col2.metric(
                "Total Frames Processed",
                report["total_frames"]
            )

            col3.metric(
                "Final Currently Tracked",
                report["currently_tracked"]
            )

            # =================================
            # FINAL OBJECT & PEOPLE COUNTING
            # =================================

            st.divider()

            st.subheader(
                "📊 Object & People Counting"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Unique Objects",
                report["total_unique_objects"]
            )

            col2.metric(
                "People Detected",
                report["people_detected"]
            )

            col3.metric(
                "Total Objects Detected",
                report["total_objects_detected"]
            )

            # =================================
            # PEOPLE CURRENTLY VISIBLE
            # =================================

            st.metric(
                "People Currently Visible",
                report["people_currently_visible"]
            )

            # =================================
            # FINAL CROWD MONITORING
            # =================================

            st.subheader(
                "👥 Crowd Monitoring"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Crowd Status",
                report["crowd_status"]
            )

            col2.metric(
                "Crowd Capacity Used",
                f'{report["crowd_percentage"]}%'
            )

            st.caption(
                f'Maximum Crowd Capacity: {report["crowd_limit"]}'
            )

            # =================================
            # INDIVIDUAL OBJECT COUNTS
            # =================================

            st.markdown(
                "### Individual Object Counts"
            )

            if report["object_counts"]:

                object_count_df = pd.DataFrame(
                    [
                        {
                            "Object": label,
                            "Count": count
                        }

                        for label, count
                        in report["object_counts"].items()
                    ]
                )

                st.dataframe(
                    object_count_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No other objects detected."
                )

            # =================================
            # OBJECT DETECTION HISTORY
            # =================================

            st.divider()

            st.subheader(
                "📋 Object Detection History"
            )

            if report["detection_history"]:

                history_df = pd.DataFrame(
                    report["detection_history"]
                )

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                history_df = pd.DataFrame(
                    columns=[
                        "Tracking ID",
                        "Object",
                        "Confidence"
                    ]
                )

                st.info(
                    "No objects were detected during this recording."
                )

            # =================================
            # DOWNLOAD DETECTION REPORTS
            # =================================

            st.divider()

            st.subheader(
                "📥 Download Detection Reports"
            )

            # =================================
            # HISTORY CSV
            # =================================

            history_csv = history_df.to_csv(
                index=False
            ).encode("utf-8")

            # =================================
            # SUMMARY CSV
            # =================================

            summary_df = pd.DataFrame([{

                "Total Unique Objects":
                    report["total_unique_objects"],

                "Total Frames Processed":
                    report["total_frames"],

                "Final Currently Tracked":
                    report["currently_tracked"],

                "People Detected":
                    report["people_detected"],

                "People Currently Visible":
                    report["people_currently_visible"],

                "Crowd Status":
                    report["crowd_status"],

                "Crowd Capacity Used (%)":
                    report["crowd_percentage"],

                "Maximum Crowd Capacity":
                    report["crowd_limit"],

                "Total Objects Detected":
                    report["total_objects_detected"]

            }])

            summary_csv = summary_df.to_csv(
                index=False
            ).encode("utf-8")

            # =================================
            # DOWNLOAD BUTTONS
            # =================================

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    label="📄 Download Detection History",
                    data=history_csv,
                    file_name="webcam_detection_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:

                st.download_button(
                    label="📊 Download Detection Summary",
                    data=summary_csv,
                    file_name="webcam_detection_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # =================================
    # DISPLAY WEBCAM STATISTICS
    # =================================

    show_webcam_results()
