# Connecting to the Lumora VPN

## Which VPN client do I need?

Lumora Systems uses the VPN client "TunnelKite".
It can be installed on Windows and macOS from the internal software portal "AppDock".
Linux users must use the WireGuard configuration file provided by the IT helpdesk.

## First-Time Setup

1. Install TunnelKite from AppDock.
2. Open TunnelKite and enter the server address `vpn.lumora.internal`.
3. Log in with your Lumora account and confirm the login in the "LumoraAuth" mobile app.
4. Choose the gateway closest to you: "FRA-1" (Frankfurt) or "LIS-2" (Lisbon).

## Common Problems

If the connection drops every few minutes, switch the protocol in TunnelKite settings from "Auto" to "TCP".
If you see error code TK-407, your device certificate has expired. Open AppDock and run "Renew Device Certificate".
The VPN is required for accessing the internal wiki, TimeHarbor and the build servers, but not for email.
