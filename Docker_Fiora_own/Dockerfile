FROM ubuntu:24.04

WORKDIR /app

RUN apt update
RUN apt upgrade -y
RUN apt install -y git
RUN apt install -y build-essential

RUN apt install -y curl
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt upgrade -y
# Create the environment:
RUN apt install -y python3.11
RUN apt install -y python3.11-venv
RUN apt install -y python3.11-dev
RUN apt install -y libxrender1
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Demonstrate the environment is activated:





RUN git clone https://github.com/Micholms/ms_model_fiora.git
RUN mv ms_model_fiora fiora
RUN ls
WORKDIR fiora

RUN pip install .

WORKDIR /app/



RUN pip install pandas streamlit
COPY files2script_webb.py .

RUN mkdir -p .streamlit/
COPY streamlit_config/ .streamlit/

WORKDIR fiora
COPY checkpoint_all_wells_mona_ft_rounded.best.pt ./
COPY checkpoint_all_wells_mona_ft_rounded.best_state.pt ./
COPY checkpoint_all_wells_mona_ft_rounded.best_params.json ./

COPY fiora_run.py ./
COPY fiora-predict ./scipts/
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "fiora_run.py", "--server.port=8501", "--server.address=0.0.0.0","--server.maxUploadSize=2000"]


