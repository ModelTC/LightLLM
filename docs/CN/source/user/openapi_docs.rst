OpenApi docs
=================================

下面的文档由openapi自动生成，在使用Lightllm部署完以后，使用  ``host:port/docs``  就可以打开

.. raw:: html

    <html>
    <head>
    <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css">
    <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <title>FastAPI - Swagger UI</title>
    <style>
        .info { display: none; }
        #swagger-ui {
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
        }
    </style>
    </head>
    <body>
    <div id="swagger-ui">
    </div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
    <!-- `SwaggerUIBundle` is now available on the page -->
    <script>
        const ui = SwaggerUIBundle({
            url: '../_static/openapi.json',
            "dom_id": "#swagger-ui",
            "layout": "BaseLayout",
            "deepLinking": true,
            "showExtensions": false,
            "showCommonExtensions": true,
            oauth2RedirectUrl: window.location.origin + '/docs/oauth2-redirect',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                    ],
                })
    </script>
    </body>
    </html>
