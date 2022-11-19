FROM tiangolo/uwsgi-nginx:python3.6
WORKDIR /flask

COPY . .
RUN pip install numpy
RUN pip install  -r requirements.txt

CMD gunicorn app:app --bind 0.0.0.0:$PORT --reload
