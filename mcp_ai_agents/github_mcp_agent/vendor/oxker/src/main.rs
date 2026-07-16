#[tokio::main]
async fn main() {
    oxker::setup_tracing();
    oxker::run().await;
}
