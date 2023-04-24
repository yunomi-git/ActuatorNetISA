FROM pytorch/pytorch

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir data

COPY . .

ENTRYPOINT [ "python3", "-u", "./dockerMain.py"]