"""
Generate a self-signed TLS certificate for LAN HTTPS access.

Required for phone mic access — browsers block getUserMedia on plain HTTP
for non-localhost origins.

Run once:
    venv-pipecat\\Scripts\\python demo\\gen_cert.py

Creates:
    demo/certs/cert.pem
    demo/certs/key.pem

On your phone: open https://<windows-ip>:8443, tap "Advanced" → "Proceed".
You only need to accept the warning once per device.
"""
import ipaddress
import socket
import datetime
from pathlib import Path

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

CERTS_DIR = Path(__file__).parent / "certs"
CERT_FILE = CERTS_DIR / "cert.pem"
KEY_FILE  = CERTS_DIR / "key.pem"


def get_all_ips() -> list:
    """Return all non-loopback IPv4 addresses on this machine."""
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ":" not in ip and ip != "127.0.0.1" and not ip.startswith("169.254"):
                if ip not in ips:
                    ips.append(ip)
    except Exception:
        pass
    # Fallback: at least get the default-route IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip not in ips:
            ips.append(ip)
    except Exception:
        pass
    return ips or ["127.0.0.1"]


def generate():
    CERTS_DIR.mkdir(parents=True, exist_ok=True)

    all_ips = get_all_ips()
    print(f"Generating self-signed cert covering IPs: {', '.join(all_ips)}")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Rammas DEWA Assistant"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DEWA"),
    ])

    san_entries = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    for ip in all_ips:
        san_entries.append(x509.IPAddress(ipaddress.IPv4Address(ip)))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    KEY_FILE.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"[OK] cert.pem  ->  {CERT_FILE}")
    print(f"[OK] key.pem   ->  {KEY_FILE}")
    print()
    print("Next step:  venv-pipecat\\Scripts\\python demo\\web_server.py")
    print("Then open one of these on your phone:")
    for ip in all_ips:
        print(f"  https://{ip}:8443")


if __name__ == "__main__":
    generate()
