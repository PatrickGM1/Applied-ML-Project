# Applied Machine Learning project

## Team members:

    - Patrick Gheba
    - Luca Serban
    - Ana-Maria Izbas
    - George Tutui

## Project idea:

    Fake News Detection using the LIAR dataset.
    Dataset: https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset

## Website for deployment:

https://aml.guba.dev/

## Nginx deployment under /demo

If the app is exposed behind nginx at https://aml.guba.dev/demo, the proxy must strip the /demo/ prefix before forwarding traffic to FastAPI. A working server block is included at deploy/nginx/aml.guba.dev.conf.
