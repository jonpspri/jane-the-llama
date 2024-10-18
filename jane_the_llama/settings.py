import warnings

from pydantic_settings import BaseSettings, SettingsConfigDict

warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        module="pydantic_settings"
        )

class JaneSettings(BaseSettings):
    model_config = SettingsConfigDict(
            env_prefix='JANE_',
            secrets_dir=('./secrets', '/run/secrets'),
            )

    milvus_host: str = 'localhost'
    milvus_port: int = 19530
    redis_host: str = 'localhost'
    redis_port: int = 6379

    milvus_token: str
    watsonx_apikey: str
    watsonx_project_id: str

    @property
    def milvus_url(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

