from __future__ import annotations

import uvicorn

from shirtrip.api.app import create_app
from shirtrip.config.settings import Settings


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
