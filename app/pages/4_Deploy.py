from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Deploy")
    st.caption("Containerize and deploy ClusterLens to Kubernetes.")

    st.subheader("Docker")
    st.code(
        "\n".join(
            [
                "docker build -t clusterlens:local .",
                "docker run --rm -p 8501:8501 clusterlens:local",
            ]
        ),
        language="bash",
    )

    st.subheader("Kubernetes")
    st.write("Update the image in `k8s/deployment.yaml`, then apply manifests:")
    st.code(
        "\n".join(
            [
                "kubectl apply -f k8s/namespace.yaml",
                "kubectl apply -f k8s/deployment.yaml",
                "kubectl apply -f k8s/service.yaml",
                "# Optional: requires an ingress controller",
                "kubectl apply -f k8s/ingress.yaml",
            ]
        ),
        language="bash",
    )

    st.info("For production, add auth (Basic/OIDC), resource limits, and persistent storage for uploaded files if needed.")


if __name__ == "__main__":
    main()

