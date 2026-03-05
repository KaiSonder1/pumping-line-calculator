Cryogenic Pumping Line Calculator (public link via Streamlit Community Cloud)
==========================================================================

What this is
------------
A small web app (Streamlit) that implements the equations from Geoffrey Nunes, Jr.,
"Pumps and Plumbing" (chapter you photographed), to propagate pressure through
a multi-section pumping line with temperature gradients.

Files
-----
- app.py             : the web UI
- nunes_pumping.py   : the solver (paper-faithful equations)
- requirements.txt   : python packages
- .streamlit/config.toml : disables error tracebacks + attempts to disable telemetry

How to run locally
------------------
1) Install Python
2) In this folder run:
   pip install -r requirements.txt
   streamlit run app.py

How to deploy for a public link
-------------------------------
Use Streamlit Community Cloud:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
