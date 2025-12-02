FROM postgis/postgis:15-3.4

# ติดตั้ง pgvector extension
RUN apt-get update && \
    apt-get install -y postgresql-15-pgvector && \
    rm -rf /var/lib/apt/lists/*

CMD ["postgres", "-c", "shared_preload_libraries=vector"]
