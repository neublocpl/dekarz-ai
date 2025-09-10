# Dekarz AI

## Getting Started

Follow these instructions to get the development environment up and running.

### Prerequisites

-   [Docker](https://www.docker.com/get-started) and Docker Compose
-   `make`
    -   **Windows:** Install via [Chocolatey](https://chocolatey.org/) (`choco install make`) or use a shell like Git Bash.
    -   **macOS/Linux:** Usually pre-installed.

### Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd dekarz-ai
    ```

2.  **Build and Run Services**
    Run the following command. This will create a `.env` file from the example, create necessary data directories, build the Docker images, and start all services (API, worker, Redis, PostgreSQL).
    ```bash
    make install
    ```

3.  **Initialize the Database**
    The services must be running before this step. Connect to the PostgreSQL container and create the `jobs` table.

    a. In the postgres container, `psql` shell, run the following SQL script:
    ```sql
    CREATE TYPE public.job_status AS ENUM ('PENDING', 'IN_PROGRESS', 'SUCCESS', 'FAILURE');

    CREATE TABLE public.jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        status public.job_status NOT NULL DEFAULT 'PENDING',
        result JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    ```
    c. Exit the `psql` shell by typing `\q`.

### Usage

1.  **Add a File**
    Place your input files (e.g., `roof_01.pdf`) into the `data/files/input/` directory on your local machine.

2.  **Start a Transformation Job**
    Use API client to send a POST request to the `/transform` endpoint. The URL for the file must point to the static path served by the API.

    ```bash
    curl -X POST "http://localhost:8080/transform" \
    -H "Content-Type: application/json" \
    -d '{"file_url": "http://localhost:8080/static/input/roof_01.pdf"}'
    ```
    The API will respond with a `job_id`.

3.  **Check Job Status**
    Use the `job_id` from the previous step to check the status and result of the job.

    ```bash
    curl http://localhost:8080/transform/<your_job_id>
    ```