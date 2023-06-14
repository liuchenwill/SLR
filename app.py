import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from aiortc.contrib.media import MediaRecorder
import streamlit.components.v1 as components
from streamlit_tensorboard import st_tensorboard

from predication import predict
import os

from real import play_webcam
from tool import image_to_video

st.set_page_config(
    page_title="Sign Language Recognition System",
    page_icon="https://s1.ax1x.com/2023/05/13/p96s11O.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/liuchenwill',
        'Report a bug': "https://github.com/liuchenwill",
        'About': "# SLR by lcc"
    }
)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Sign Language Recognition System")

with st.sidebar:
    st.header("配置")
    model_options = ("r2+1d_100", "r3d_100", "LSTM_100", "r2+1d_500", "r3d_500", "LSTM_500")
    selected_model = st.selectbox(
        label="选择使用的模型:",
        options=model_options,
    )
    page_weight = (
        os.listdir("weight/"+str(selected_model))
    )
    page_weight.sort(reverse=True)
    selected_weight = st.selectbox(
        label="选择训练的权重",
        options=page_weight,
    )
    video_style = st.selectbox(
        label="选择视频",
        options=("上传视频", "录制视频", "CSL测试集", "实时识别")
    )

if video_style == "上传视频":
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['mp4', 'mp4v'])
    flag = False
elif video_style == "录制视频":
    flag = True
    uploaded_file = None
elif video_style == "实时识别":
    play_webcam(selected_model, selected_weight)
    flag = False
    uploaded_file = None
else:
    flag = False
    video_options=[f"{i:03}.mp4" for i in range(500)]
    selected_video = st.sidebar.selectbox(
        label="Choose a file",
        options=video_options
    )
    uploaded_file = "../SLR_Dataset/CSL_Isolated/ptov/" + selected_video

with st.sidebar:

    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    # flag = st.checkbox('录制视频')

    # def recorder_factory():
    #     return MediaRecorder("data/videos/1022chen.mp4")

    # if flag:
    #     webrtc_streamer(
    #         key="demo",
    #         # media_stream_constraints={"video": True, "audio": True},
    #         in_recorder_factory=recorder_factory,
    #     )

    class VideoTransformer(VideoTransformerBase):

        def __init__(self):
            self.cnt = 0
        def transform(self, frame):

            img = frame.to_ndarray(format="bgr24")
            frame.to_image().save('data/images/%06d.jpg'%self.cnt)
            self.cnt += 1

            return img
    if flag:
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)




if (uploaded_file is not None) | flag:
    if (flag == False) & (video_style == "上传视频"):
        is_valid = True
        with st.spinner(text='资源加载中...'):
            st.sidebar.video(uploaded_file)
            with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
    else:
        is_valid = True
        if(video_style=="CSL测试集"):
            st.sidebar.video(uploaded_file)
else:
    is_valid = False
    st.write(selected_model + ":lemon: 网络的展示")
    # logdir = "logs/{}".format(selected_model)
    # print(logdir)
    # cnt = 0
    # cnt += 1
    # st_tensorboard(logdir=logdir, port=6006+cnt, width=680)

    tab1, tab2, tab3, tab4 = st.tabs(["acc曲线图", "loss曲线图", "网络可视化", "Confusion Matrix"])

    with tab1:

        df = pd.read_csv('logs/csv/{}_acc.csv'.format(selected_model))
        df.index = df.index + 1
        st.line_chart(df)


        image = Image.open('logs/img/{}_acc.png'.format(selected_model))
        st.image(image, caption='acc曲线图')

    with tab2:

        df = pd.read_csv('logs/csv/{}_loss.csv'.format(selected_model))
        df.index = df.index + 1
        st.line_chart(df)

        image = Image.open('logs/img/{}_loss.png'.format(selected_model))
        st.image(image, caption='loss曲线图')

    with tab3:
        display = open('logs/net/{}.onnx.svg'.format(selected_model), 'r', encoding='utf-8')
        source_code = display.read()
        components.html(source_code, height=1000, scrolling=True)

    with tab4:
        image = Image.open('logs/confusion_matrix/{}_confmat.png'.format(selected_model))
        st.image(image, caption='混淆矩阵')


xianshi = False

if is_valid:
    print('valid...')
    st1, st2 = st.columns(2)
    with st1:
        agree = st.checkbox('移除视频背景')
        if agree:

            st.write("移除背景模型运行在CPU上，可能需要几分钟！")
    with st2:
        centercrop = st.checkbox('裁剪居中')

    if st.button(':cake:开始识别'):


        # labels = predict(video_path)
        # st.write("Prediction index", labels[0], ", Prediction word: ", labels[1])

        # video_file = open(video_path, 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes)

        if flag:    # 录制视频
            image_path = "data/images"

            import uuid
            uuid_str = uuid.uuid4().hex
            tmp_file_name = 'tmpfile_%s.mp4' % uuid_str

            media_path = "data/videos/"+tmp_file_name
            image_to_video(sample_dir=image_path, video_name=media_path)
            ida, prediction = predict(tmp_file_name, selected_model, selected_weight, agree, video_style, centercrop)
            xianshi = True
        else:
            if video_style == "上传视频":
                ida, prediction = predict(uploaded_file.name, selected_model, selected_weight, agree, video_style, centercrop)
            else:
                ida, prediction = predict(selected_video, selected_model, selected_weight, agree, video_style, centercrop)
        st.write(":lemon: Prediction index", ida)
        st.write(":cherries: Prediction word: ", prediction[0][0])
        st.title("TOP-5 Likely Word :beers:")
        df = pd.DataFrame(data=np.zeros((5, 2)),
                          columns=['Word', 'Confidence Level'],
                          index=np.linspace(1, 5, 5, dtype=int))

        for idx, p in enumerate(prediction):
            ans = str(p[0])
            word = ans.split("（")
            link = 'https://www.spreadthesign.com/zh.hans.cn/search/?q=' + str(word[0])
            df.iloc[idx, 0] = f'<a href="{link}" target="_blank">{ans}</a>'
            df.iloc[idx, 1] = p[1]
        st.write(df.to_html(escape=False), unsafe_allow_html=True)

        st.balloons()


    else:
        st.write(f':point_up: Click :point_up:')

with st.sidebar:

    if flag & xianshi:
        st.video("data/videos/"+tmp_file_name)
        imgPath = "data/images/"
        ls = os.listdir(imgPath)
        for i in ls:
            c_path = os.path.join(imgPath, i)
            os.remove(c_path)

    st.markdown("---")
    st.markdown(
        '<h6>&nbsp<img src="https://s1.ax1x.com/2023/05/13/p96s11O.png" alt="logo" height="16">&nbsp by <a href="https://github.com/liuchenwill">@liuchenchen</a></h6>',
        unsafe_allow_html=True,
    )
