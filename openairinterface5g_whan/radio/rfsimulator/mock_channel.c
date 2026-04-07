/*
 * Mock Channel Module for OAI
 * 
 * This module implements a bypass channel that simply forwards data
 * from gNB to UE without any channel modeling.
 * 
 * Purpose: Test OAI ↔ Channel ↔ Sionna integration path
 * 
 * Author: Minsoo Kim
 * Date: 2024-07-16
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>

#include "rfsimulator.h"
#include "openair1/PHY/TOOLS/tools_defs.h" // c16_t, cf_t, channel_desc_t 등

// Mock channel configuration
#define MOCK_CHANNEL_PORT_GNB 6017  // Port for gNB connection (same as gNB)
#define MOCK_CHANNEL_PORT_UE  6018  // Port for UE connection (same as UE)
#define MAX_BUFFER_SIZE 8192
#define MAX_CLIENTS 10

typedef struct {
    int gnb_socket;
    int ue_socket;
    int gnb_client_socket;
    int ue_client_socket;
    pthread_t gnb_thread;
    pthread_t ue_thread;
    int running;
} mock_channel_state_t;

static mock_channel_state_t mock_state = {0};

// Forward declaration
static void* gnb_handler_thread(void* arg);
static void* ue_handler_thread(void* arg);
static int setup_server_socket(int port);
static int connect_to_gnb_server(void);
static int accept_client_connection(int server_socket);
static void cleanup_sockets(void);

/**
 * Initialize mock channel
 * 
 * @return 0 on success, -1 on failure
 */
int mock_channel_init(void) {
    printf("[MOCK_CHANNEL] Initializing mock channel...\n");
    
    // Setup UE server socket (UE connects to us)
    mock_state.ue_socket = setup_server_socket(MOCK_CHANNEL_PORT_UE);
    if (mock_state.ue_socket < 0) {
        printf("[MOCK_CHANNEL] Failed to setup UE server socket\n");
        return -1;
    }
    
    // Initialize gNB socket to -1 (will connect as client)
    mock_state.gnb_socket = -1;
    
    mock_state.running = 1;
    
    printf("[MOCK_CHANNEL] Mock channel initialized successfully\n");
    printf("[MOCK_CHANNEL] Will connect to gNB server on port %d\n", MOCK_CHANNEL_PORT_GNB);
    printf("[MOCK_CHANNEL] UE server listening on port %d\n", MOCK_CHANNEL_PORT_UE);
    
    return 0;
}

/**
 * Start mock channel threads
 * 
 * @return 0 on success, -1 on failure
 */
int mock_channel_start(void) {
    printf("[MOCK_CHANNEL] Starting mock channel threads...\n");
    
    // Start gNB handler thread
    if (pthread_create(&mock_state.gnb_thread, NULL, gnb_handler_thread, NULL) != 0) {
        printf("[MOCK_CHANNEL] Failed to create gNB handler thread\n");
        return -1;
    }
    
    // Start UE handler thread
    if (pthread_create(&mock_state.ue_thread, NULL, ue_handler_thread, NULL) != 0) {
        printf("[MOCK_CHANNEL] Failed to create UE handler thread\n");
        mock_state.running = 0;
        pthread_join(mock_state.gnb_thread, NULL);
        return -1;
    }
    
    printf("[MOCK_CHANNEL] Mock channel threads started successfully\n");
    return 0;
}

/**
 * Stop mock channel
 */
void mock_channel_stop(void) {
    printf("[MOCK_CHANNEL] Stopping mock channel...\n");
    
    mock_state.running = 0;
    
    // Wait for threads to finish
    if (mock_state.gnb_thread) {
        pthread_join(mock_state.gnb_thread, NULL);
    }
    if (mock_state.ue_thread) {
        pthread_join(mock_state.ue_thread, NULL);
    }
    
    cleanup_sockets();
    
    printf("[MOCK_CHANNEL] Mock channel stopped\n");
}

/**
 * gNB handler thread - connects to gNB as client and forwards data to UE
 */
static void* gnb_handler_thread(void* arg) {
    printf("[MOCK_CHANNEL] gNB handler thread started\n");
    
    while (mock_state.running) {
        // Connect to gNB as client
        mock_state.gnb_client_socket = connect_to_gnb_server();
        if (mock_state.gnb_client_socket < 0) {
            if (mock_state.running) {
                printf("[MOCK_CHANNEL] Failed to connect to gNB server\n");
            }
            sleep(1); // Wait before retry
            continue;
        }
        
        printf("[MOCK_CHANNEL] gNB connected\n");
        
        // Wait for UE to connect
        while (mock_state.running && mock_state.ue_client_socket <= 0) {
            usleep(100000); // 100ms
        }
        
        if (!mock_state.running) {
            close(mock_state.gnb_client_socket);
            break;
        }
        
        // Forward data from gNB to UE
        char buffer[MAX_BUFFER_SIZE];
        ssize_t bytes_received;
        
        while (mock_state.running && (bytes_received = recv(mock_state.gnb_client_socket, buffer, sizeof(buffer), 0)) > 0) {
            printf("[MOCK_CHANNEL] Forwarding %zd bytes from gNB to UE\n", bytes_received);
            
            ssize_t bytes_sent = send(mock_state.ue_client_socket, buffer, bytes_received, 0);
            if (bytes_sent != bytes_received) {
                printf("[MOCK_CHANNEL] Failed to forward data to UE\n");
                break;
            }
        }
        
        printf("[MOCK_CHANNEL] gNB disconnected\n");
        close(mock_state.gnb_client_socket);
        mock_state.gnb_client_socket = -1;
    }
    
    printf("[MOCK_CHANNEL] gNB handler thread stopped\n");
    return NULL;
}

/**
 * UE handler thread - receives data from UE and forwards to gNB
 */
static void* ue_handler_thread(void* arg) {
    printf("[MOCK_CHANNEL] UE handler thread started\n");
    
    while (mock_state.running) {
        // Accept UE connection
        mock_state.ue_client_socket = accept_client_connection(mock_state.ue_socket);
        if (mock_state.ue_client_socket < 0) {
            if (mock_state.running) {
                printf("[MOCK_CHANNEL] Failed to accept UE connection\n");
            }
            continue;
        }
        
        printf("[MOCK_CHANNEL] UE connected\n");
        
        // Wait for gNB to connect
        while (mock_state.running && mock_state.gnb_client_socket <= 0) {
            usleep(100000); // 100ms
        }
        
        if (!mock_state.running) {
            close(mock_state.ue_client_socket);
            break;
        }
        
        // Forward data from UE to gNB
        char buffer[MAX_BUFFER_SIZE];
        ssize_t bytes_received;
        
        while (mock_state.running && (bytes_received = recv(mock_state.ue_client_socket, buffer, sizeof(buffer), 0)) > 0) {
            printf("[MOCK_CHANNEL] Forwarding %zd bytes from UE to gNB\n", bytes_received);
            
            ssize_t bytes_sent = send(mock_state.gnb_client_socket, buffer, bytes_received, 0);
            if (bytes_sent != bytes_received) {
                printf("[MOCK_CHANNEL] Failed to forward data to gNB\n");
                break;
            }
        }
        
        printf("[MOCK_CHANNEL] UE disconnected\n");
        close(mock_state.ue_client_socket);
        mock_state.ue_client_socket = -1;
    }
    
    printf("[MOCK_CHANNEL] UE handler thread stopped\n");
    return NULL;
}

/**
 * Connect to gNB server as client
 */
static int connect_to_gnb_server(void) {
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
        printf("[MOCK_CHANNEL] Failed to create client socket\n");
        return -1;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(MOCK_CHANNEL_PORT_GNB);
    
    if (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        printf("[MOCK_CHANNEL] Failed to connect to gNB server on port %d\n", MOCK_CHANNEL_PORT_GNB);
        close(client_socket);
        return -1;
    }
    
    printf("[MOCK_CHANNEL] Connected to gNB server on port %d\n", MOCK_CHANNEL_PORT_GNB);
    return client_socket;
}

/**
 * Setup server socket
 */
static int setup_server_socket(int port) {
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        printf("[MOCK_CHANNEL] Failed to create server socket\n");
        return -1;
    }
    
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        printf("[MOCK_CHANNEL] Failed to set socket options\n");
        close(server_socket);
        return -1;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        printf("[MOCK_CHANNEL] Failed to bind server socket to port %d\n", port);
        close(server_socket);
        return -1;
    }
    
    if (listen(server_socket, MAX_CLIENTS) < 0) {
        printf("[MOCK_CHANNEL] Failed to listen on server socket\n");
        close(server_socket);
        return -1;
    }
    
    return server_socket;
}

/**
 * Accept client connection with timeout
 */
static int accept_client_connection(int server_socket) {
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    
    // Set socket to non-blocking mode
    int flags = fcntl(server_socket, F_GETFL, 0);
    fcntl(server_socket, F_SETFL, flags | O_NONBLOCK);
    
    // Try to accept with timeout
    int client_socket = -1;
    for (int i = 0; i < 100 && mock_state.running; i++) { // 10 second timeout
        client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
        if (client_socket >= 0) {
            break;
        }
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            break;
        }
        usleep(100000); // 100ms
    }
    
    // Restore blocking mode
    fcntl(server_socket, F_SETFL, flags);
    
    if (client_socket < 0) {
        return -1;
    }
    
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
    printf("[MOCK_CHANNEL] Client connected from %s:%d\n", client_ip, ntohs(client_addr.sin_port));
    
    return client_socket;
}

/**
 * Cleanup sockets
 */
static void cleanup_sockets(void) {
    if (mock_state.gnb_client_socket >= 0) {
        close(mock_state.gnb_client_socket);
        mock_state.gnb_client_socket = -1;
    }
    if (mock_state.ue_client_socket >= 0) {
        close(mock_state.ue_client_socket);
        mock_state.ue_client_socket = -1;
    }
    if (mock_state.gnb_socket >= 0) {
        close(mock_state.gnb_socket);
        mock_state.gnb_socket = -1;
    }
    if (mock_state.ue_socket >= 0) {
        close(mock_state.ue_socket);
        mock_state.ue_socket = -1;
    }
}

/**
 * Main function for standalone testing
 */
#ifdef MOCK_CHANNEL_STANDALONE
void handle_sigint(int sig) {
    mock_state.running = 0;
}

int main(int argc, char* argv[]) {
    printf("[MOCK_CHANNEL] Starting mock channel in standalone mode\n");
    
    // Setup signal handler for graceful shutdown
    signal(SIGINT, handle_sigint);
    
    if (mock_channel_init() != 0) {
        printf("[MOCK_CHANNEL] Failed to initialize mock channel\n");
        return -1;
    }
    
    if (mock_channel_start() != 0) {
        printf("[MOCK_CHANNEL] Failed to start mock channel\n");
        return -1;
    }
    
    printf("[MOCK_CHANNEL] Mock channel running. Press Ctrl+C to stop.\n");
    
    // Wait for shutdown signal
    while (mock_state.running) {
        sleep(1);
    }
    
    mock_channel_stop();
    printf("[MOCK_CHANNEL] Mock channel shutdown complete\n");
    
    return 0;
}
#endif 