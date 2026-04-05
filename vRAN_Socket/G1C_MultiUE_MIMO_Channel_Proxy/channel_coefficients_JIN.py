"""
Class for sampling channel impulse responses following 3GPP TR38.901
specifications and giving LSPs and rays.
"""



import tensorflow as tf
from tensorflow import sin, cos, acos
from sionna.phy import PI, SPEED_OF_LIGHT, config
from sionna.phy.utils import expand_to_rank
from sionna.phy.block import Object
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
import time
import timeit
import cupy as cp
import subprocess

##################################
# GPU 정보 확인
##################################

def gpu_info_check():
    print("=== GPU 정보 확인 ===")
    
    # TensorFlow로 GPU 목록 확인
    print("\n1. TensorFlow GPU 정보:")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("TensorFlow가 인식한 GPU가 없습니다.")
    else:
        for gpu in gpus:
            print(f"발견된 GPU: {gpu.name}")
            try:
                # GPU 메모리 설정 확인
                memory_growth = tf.config.experimental.get_memory_growth(gpu)
                print(f"메모리 자동 증가 설정: {memory_growth}")
            except:
                print("메모리 설정 정보를 가져올 수 없습니다.")
    
    # nvidia-smi 명령어로 자세한 정보 확인
    print("\n2. nvidia-smi 정보:")
    try:
        nvidia_smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free,memory.total',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        print("GPU 인덱스, 이름, 사용 중인 메모리(MB), 가용 메모리(MB), 전체 메모리(MB)")
        for line in nvidia_smi.strip().split('\n'):
            print(line)
    except:
        print("nvidia-smi 명령을 실행할 수 없습니다.")

def print_vram_usage():
    """현재 GPU VRAM 사용량을 출력"""
    mem_info = tf.config.experimental.get_memory_info('/GPU:0')
    used_memory = mem_info['current'] / (1024**3)  # GB 단위 변환
    peak_memory = mem_info['peak'] / (1024**3)  # GB 단위 변환
    print(f"현재 GPU VRAM 사용량: {used_memory:.2f} GB / {peak_memory:.2f} GB")
    print("\n")

def get_tensor_memory_usage(tensor):
    """ 주어진 TensorFlow 텐서가 GPU 메모리에서 차지하는 용량(GB)을 계산 """
    num_elements = tf.size(tensor).numpy()  # 텐서의 총 요소 개수
    dtype_size = tensor.dtype.size  # 자료형 크기 (바이트)
    total_bytes = num_elements * dtype_size  # 전체 메모리 사용량 (바이트)
    total_gb = round(total_bytes / (1024 ** 3), 4)  # GB 단위 변환
    return total_gb

##################################
# Binary Mask 생성
##################################

def random_binary_mask_tf(n, k=3):
    """
    n개 중에서 랜덤하게 k개를 1로 설정하는 이진 마스크를 생성합니다.
    
    Args:
        n: 전체 요소 수
        k: 1로 설정할 요소 수 (기본값: 3)
    
    Returns:
        shape이 [n]인 이진 마스크 텐서
    """
    # 모두 0으로 초기화된 텐서 생성
    mask = tf.zeros([n], dtype=tf.int32)
    
    # 랜덤하게 인덱스 선택
    indices = tf.random.shuffle(tf.range(n))[:k]
    
    # 선택된 인덱스 위치에 1 설정
    updates = tf.ones([k], dtype=tf.int32)
    mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=-1), updates)
    
    return mask

def sample_unique_indices(A, a):
    """
    0 ~ A-1 중에서 중복 없이 a개를 랜덤하게 선택
    Args:
        A (int): 전체 후보 수 (예: 128)
        a (int): 선택할 개수 (예: 20)
    Returns:
        tf.Tensor: shape=(a,)인 정수형 텐서
    """
    sampled = tf.random.shuffle(tf.range(A))[:a]
    return tf.sort(sampled)


def random_binary_mask_tf_complex64(n, k=3):
    """
    n개 중에서 랜덤하게 k개를 1로 설정하는 이진 마스크를 생성합니다.
    
    Args:
        n: 전체 요소 수
        k: 1로 설정할 요소 수 (기본값: 3)
    
    Returns:
        shape이 [n]인 이진 마스크 텐서
    """
    # 모두 0으로 초기화된 텐서 생성
    mask = tf.zeros([n], dtype=tf.complex64)
    
    # 랜덤하게 인덱스 선택
    indices = tf.random.shuffle(tf.range(n))[:k]
    
    # 선택된 인덱스 위치에 1 설정
    updates = tf.ones([k], dtype=tf.complex64)
    mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=-1), updates)
    
    return mask

def delay2freq(h_delay):
    h_freq = tf.signal.fft(h_delay)
    return h_freq



##################################
# Precoder and Detector 
##################################

def initialize_default_precoder_detector(B, N_BS, N_UE, N_t, N_r, N_s, N_sym, N_fft, dtype=tf.complex64):
    # 1. Create identity matrix [N_s, N_s] for both precoder and detector
    eye = tf.eye(N_s, dtype=dtype)  # [N_s, N_s]

    # 2. Expand and broadcast for W_precoding: [B, N_BS, N_UE, N_t, N_s, N_sym, N_fft]
    # Only upper-left I_{N_s} in N_t x N_s: pad identity if N_t > N_s
    W_matrix = tf.concat([
        eye,  # shape [N_s, N_s]
        tf.zeros([N_t - N_s, N_s], dtype=dtype)
    ], axis=0) if N_t > N_s else eye[:N_t, :]  # shape [N_t, N_s]

    W_matrix = tf.reshape(W_matrix, [1, 1, 1, N_t, N_s])  # [1,1,1,N_t,N_s]
    W_matrix = tf.broadcast_to(W_matrix, [B, N_BS, N_UE, N_t, N_s])  # [B,N_BS,N_UE,N_t,N_s]
    W_precoding = tf.expand_dims(W_matrix, axis=-1)  # [B,N_BS,N_UE,N_t,N_s,1]
    W_precoding = tf.tile(W_precoding, [1,1,1,1,1,N_sym])  # [B,N_BS,N_UE,N_t,N_s,N_sym]
    W_precoding = tf.expand_dims(W_precoding, axis=-1)  # [B,N_BS,N_UE,N_t,N_s,N_sym,1]
    W_precoding = tf.tile(W_precoding, [1,1,1,1,1,1,N_fft])  # [B,N_BS,N_UE,N_t,N_s,N_sym,N_fft]

    # 3. Expand and broadcast for G_detector: [B, N_BS, N_UE, N_s, N_r, N_sym, N_fft]
    G_matrix = tf.concat([
        eye,  # [N_s, N_s]
        tf.zeros([N_s, N_r - N_s], dtype=dtype)
    ], axis=1) if N_r > N_s else eye[:, :N_r]  # shape [N_s, N_r]

    G_matrix = tf.reshape(G_matrix, [1, 1, 1, N_s, N_r])  # [1,1,1,N_s,N_r]
    G_matrix = tf.broadcast_to(G_matrix, [B, N_BS, N_UE, N_s, N_r])  # [B,N_BS,N_UE,N_s,N_r]
    G_detector = tf.expand_dims(G_matrix, axis=-1)  # [B,N_BS,N_UE,N_s,N_r,1]
    G_detector = tf.tile(G_detector, [1,1,1,1,1,N_sym])  # [B,N_BS,N_UE,N_s,N_r,N_sym]
    G_detector = tf.expand_dims(G_detector, axis=-1)  # [B,N_BS,N_UE,N_s,N_r,N_sym,1]
    G_detector = tf.tile(G_detector, [1,1,1,1,1,1,N_fft])  # [B,N_BS,N_UE,N_s,N_r,N_sym,N_fft]

    return W_precoding, G_detector


class Topology(Object):
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing the network topology information required
    for sampling channel impulse responses

    Parameters
    -----------
    velocities : [batch size, number of UTs], `tf.float`
        UT velocities

    moving_end : "tx" | "rx"
        Indicated which end of the channel (TX or RX) is moving

    los_aoa : [batch size, number of BSs, number of UTs], `tf.float`
        Azimuth angle of arrival of LoS path [radian]

    los_aod : [batch size, number of BSs, number of UTs], `tf.float`
        Azimuth angle of departure of LoS path [radian]

    los_zoa : [batch size, number of BSs, number of UTs], `tf.float`
        Zenith angle of arrival for of path [radian]

    los_zod : [batch size, number of BSs, number of UTs], `tf.float`
        Zenith angle of departure for of path [radian]

    los : [batch size, number of BSs, number of UTs], `tf.bool`
        Indicate for each BS-UT link if it is in LoS

    distance_3d : [batch size, number of UTs, number of UTs], `tf.float`
        Distance between the UTs in X-Y-Z space (not only X-Y plan)

    tx_orientations : [batch size, number of TXs, 3], `tf.float`
        Orientations of the transmitters, which are either BSs or UTs depending
        on the link direction [radian]

    rx_orientations : [batch size, number of RXs, 3], `tf.float`
        Orientations of the receivers, which are either BSs or UTs depending on
        the link direction [radian]
    """
    def __init__(self,  velocities,
                        moving_end,
                        los_aoa,
                        los_aod,
                        los_zoa,
                        los_zod,
                        los,
                        distance_3d,
                        tx_orientations,
                        rx_orientations):
        self.velocities = velocities
        self.moving_end = moving_end
        self.los_aoa = los_aoa
        self.los_aod = los_aod
        self.los_zoa = los_zoa
        self.los_zod = los_zod
        self.los = los
        self.tx_orientations = tx_orientations
        self.rx_orientations = rx_orientations
        self.distance_3d = distance_3d
        super().__init__()

class ChannelCoefficientsGeneratorJIN(Object):
    # pylint: disable=line-too-long
    r"""
    Sample channel impulse responses according to LSPs rays

    This class implements steps 10 and 11 from the TR 38.901 specifications,
    (section 7.5).

    Parameters
    ----------
    carrier_frequency : `float`
        Carrier frequency [Hz]

    tx_array : class:`~sionna.phy.channel.tr38901.PanelArray`
        Array used by the transmitters.
        All transmitters share the same antenna array configuration.

    rx_array : class:`~sionna.phy.channel.tr38901.PanelArray`
        Panel array used by the receivers.
        All receivers share the same antenna array configuration.

    subclustering : `bool`, (default `True`)
        Use subclustering if set to `True` (see step 11 for section 7.5 in
        TR 38.901). CDL does not use subclustering. System level models (UMa,
        UMi, RMa) do.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    num_time_samples : `int`
        Number of samples

    sampling_frequency : `float`
        Sampling frequency [Hz]

    k_factor : [batch_size, number of TX, number of RX], `tf.float`
        K-factor

    rays : `Rays`
        Rays from which to compute thr CIR

    topology : `Topology`
        Topology of the network

    c_ds : [batch size, number of TX, number of RX], `tf.float`
        Cluster DS [ns]. Only needed when subclustering is used
        (``subclustering`` set to `True`), i.e., with system level models.
        Otherwise can be set to None.
        Defaults to None.

    debug : `bool`
        If set to `True`, additional information is returned in addition to
        paths coefficients and delays: The random phase shifts (see step 10 of
        section 7.5 in TR38.901 specification), and the time steps at which the
        channel is sampled.

    Output
    ------
    h : [batch size, num TX, num RX, num paths, num RX antenna, num TX antenna, num samples], `tf.complex`
        Paths coefficients

    delays : [batch size, num TX, num RX, num paths], `tf.float`
        Paths delays [s]

    phi : [batch size, number of BSs, number of UTs, 4], `tf.float`
        Initial phases (see step 10 of section 7.5 in TR 38.901 specification).
        Last dimension corresponds to the four polarization combinations.

    sample_times : [number of time steps], `tf.float`
        Sampling time steps
    """

    def __init__(self,  carrier_frequency, subcarrier_spacing,
                        tx_array, rx_array,
                        subclustering,
                        precision=None):
        super().__init__(precision=precision)

        # Wavelength (m)
        self.carrier_frequency = carrier_frequency
        self.subcarrier_spacing = subcarrier_spacing
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT/carrier_frequency,
                                     self.rdtype)
        self._tx_array = tx_array
        self._rx_array = rx_array
        self._subclustering = subclustering

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        self._sub_cl_1_ind = tf.constant([0,1,2,3,4,5,6,7,18,19], tf.int32)
        self._sub_cl_2_ind = tf.constant([8,9,10,11,16,17], tf.int32)
        self._sub_cl_3_ind = tf.constant([12,13,14,15], tf.int32)
        self._sub_cl_delay_offsets = tf.constant([0, 1.28, 2.56], self.rdtype)

    def __call__(self, num_time_samples, sampling_frequency, k_factor, rays,
                 topology, c_ds=None, debug=False):
        # Sample times
        sample_times = (tf.range(num_time_samples,
                dtype=self.rdtype)/sampling_frequency)

        # Step 10
        phi = self._step_10(tf.shape(rays.aoa))

        # Step 11
        h, delays = self._step_11(phi, topology, k_factor, rays, sample_times,
                                                                        c_ds)

        # Return additional information if requested
        if debug:
            return h, delays, phi, sample_times

        return h, delays


    ###########################################
    # Utility functions
    ###########################################

    def _unit_sphere_vector(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, `tf.float`
            Zenith [radian]

        phi : Same shape as ``theta``, `tf.float`
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-1)
    
    def _unit_sphere_vector_Modified(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, `tf.float`
            Zenith [radian]

        phi : Same shape as ``theta``, `tf.float`
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=1)
    
    def _unit_sphere_vector_Modified2(self, theta, phi):

        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-3)

    def _forward_rotation_matrix(self, orientations):
        r"""
        Forward composite rotation matrix (7.1-4)

        Input
        ------
            orientations : [...,3], `tf.float`
                Orientation to which to rotate [radian]

        Output
        -------
        R : [...,3,3], `tf.float`
            Rotation matrix
        """
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]

        row_1 = tf.stack([cos(a)*cos(b),
            cos(a)*sin(b)*sin(c)-sin(a)*cos(c),
            cos(a)*sin(b)*cos(c)+sin(a)*sin(c)], axis=-1)

        row_2 = tf.stack([sin(a)*cos(b),
            sin(a)*sin(b)*sin(c)+cos(a)*cos(c),
            sin(a)*sin(b)*cos(c)-cos(a)*sin(c)], axis=-1)

        row_3 = tf.stack([-sin(b),
            cos(b)*sin(c),
            cos(b)*cos(c)], axis=-1)

        rot_mat = tf.stack([row_1, row_2, row_3], axis=-2)
        return rot_mat

    def _rot_pos(self, orientations, positions):
        r"""
        Rotate the ``positions`` according to the ``orientations``

        Input
        ------
        orientations : [...,3], `tf.float`
            Orientation to which to rotate [radian]

        positions : [...,3,1], `tf.float`
            Positions to rotate

        Output
        -------
        : [...,3,1], `tf.float`
            Rotated positions
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return tf.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations):
        r"""
        Reverse composite rotation matrix (7.1-4)

        Input
        ------
        orientations : [...,3], `tf.float`
            Orientations to rotate to  [radian]

        Output
        -------
        R_inv : [...,3,3], `tf.float`
            Inverse of the rotation matrix corresponding to ``orientations``
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = tf.linalg.matrix_transpose(rot_mat)
        return rot_mat_inv

    def _gcs_to_lcs(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        Input
        ------
        orientations : [...,3] of rank K, `tf.float`
            Orientations to which to rotate to [radian]

        theta : Broadcastable to the first K-1 dimensions of ``orientations``, `tf.float`
            Zenith to rotate [radian]

        phi : Same dimension as ``theta``, `tf.float`
            Azimuth to rotate [radian]

        Output
        -------
        theta_prime : Same dimension as ``theta``, `tf.float`
            Rotated zenith

        phi_prime : Same dimensions as ``theta`` and ``phi``, `tf.float`
            Rotated azimuth
        """
        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = tf.matmul(rot_inv, rho_hat)
        v1 = tf.constant([0,0,1], self.rdtype)
        v1 = tf.reshape(v1, [1]*(rot_rho.shape.rank-1)+[3])
        v2 = tf.constant([1+0j,1j,0], self.cdtype)
        v2 = tf.reshape(v2, [1]*(rot_rho.shape.rank-1)+[3])
        z = tf.matmul(v1, rot_rho)
        z = tf.clip_by_value(z, tf.constant(-1., self.rdtype),
                             tf.constant(1., self.rdtype))
        theta_prime = acos(z)
        phi_prime = tf.math.angle((tf.matmul(v2, tf.cast(rot_rho,
            self.cdtype))))
        theta_prime = tf.squeeze(theta_prime, axis=[phi.shape.rank,
            phi.shape.rank+1])
        phi_prime = tf.squeeze(phi_prime, axis=[phi.shape.rank,
            phi.shape.rank+1])

        return (theta_prime, phi_prime)

    def _compute_psi(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        Input
        ------
        orientations : [...,3], tf.float
            Orientations to which to rotate to [radian]

        theta :  Broadcastable to the first K-1 dimensions of ``orientations``, tf.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, tf.float
            Spherical position azimuth [radian]

        Output
        -------
            Psi : Same shape as ``theta`` and ``phi``, tf.float
                Displacement angle :math:`Psi`
        """
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        real = sin(c)*cos(theta)*sin(phi-a)
        real += cos(c)*(cos(b)*sin(theta)-sin(b)*cos(theta)*cos(phi-a))
        imag = sin(c)*cos(phi-a) + sin(b)*cos(c)*sin(phi-a)
        psi = tf.math.angle(tf.complex(real, imag))
        return psi

    def _l2g_response(self, f_prime, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Transform field components from LCS to GCS (7.1-11)

        Input
        ------
        f_prime : K-Dim Tensor of shape [...,2], tf.float
            Field components

        orientations : K-Dim Tensor of shape [...,3], tf.float
            Orientations of LCS-GCS [radian]

        theta : K-1-Dim Tensor with matching dimensions to ``f_prime`` and ``phi``, tf.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, tf.float
            Spherical position azimuth [radian]

        Output
        ------
            F : K+1-Dim Tensor with shape [...,2,1], tf.float
                The first K dimensions are identical to those of ``f_prime``
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = tf.stack([cos(psi), -sin(psi)], axis=-1)
        row2 = tf.stack([sin(psi), cos(psi)], axis=-1)
        mat = tf.stack([row1, row2], axis=-2)
        f = tf.matmul(mat, tf.expand_dims(f_prime, -1))
        return f

    def _step_11_get_tx_antenna_positions(self, topology):
        r"""Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_tx : [batch_size, num TXs, num TX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get BS orientations got broadcasting
        tx_orientations = topology.tx_orientations
        tx_orientations = tf.expand_dims(tx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = tf.reshape(tx_ant_pos_lcs,
            [1,1]+tx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = tf.reshape(tx_ant_pos_gcs,
            tf.shape(tx_ant_pos_gcs)[:-1])

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    def _step_11_get_rx_antenna_positions(self, topology):
        r"""Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_rx : [batch_size, num RXs, num RX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get UT orientations got broadcasting
        rx_orientations = topology.rx_orientations
        rx_orientations = tf.expand_dims(rx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = tf.reshape(rx_ant_pos_lcs,
            [1,1]+rx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = tf.reshape(rx_ant_pos_gcs,
            tf.shape(rx_ant_pos_gcs)[:-1])

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10(self, shape):
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], tf.float
            Phases for all polarization combinations
        """
        phi = config.tf_rng.uniform(
                             tf.concat([shape, [4]], axis=0),
                             minval=-PI,
                             maxval=PI,
                             dtype=self.rdtype)

        return phi

    def angle_Expansion(self, rays, fft_size, scs):
        # 1. delay indices 계산


        print("tf.shape(rays.aoa):", tf.shape(rays.aoa))
        print("tf.shape(rays.aod):", tf.shape(rays.aod))
        print("tf.shape(rays.zoa):", tf.shape(rays.zoa))
        print("tf.shape(rays.zod):", tf.shape(rays.zod))
        print("tf.shape(rays.xpr):", tf.shape(rays.xpr))
        print("tf.shape(rays.powers):", tf.shape(rays.powers))
        print("\n")

        print("tf.shape(rays.delays):", tf.shape(rays.delays))
        print("\n")

        tic = time.time()
        delayIndex = tf.cast(tf.floor(rays.delays*scs*fft_size), dtype=tf.int32)
        delayIndex = delayIndex-tf.reduce_min(delayIndex, axis=-1, keepdims=True)
        print("tf.shape(delayIndex):", tf.shape(delayIndex))
        print("========= delay indices 연산 소요시간:", round((time.time()-tic)*1000,4), "ms")
        print("\n")

        # 2. 각도들을 delay domain으로 확장
        tic = time.time()
        """
        def expand_to_delay(values):
            indices = tf.stack([
                tf.tile(tf.range(tf.shape(delayIndex)[0]), [tf.reduce_prod(tf.shape(delayIndex)[1:])]),
                tf.tile(tf.repeat(tf.range(tf.shape(delayIndex)[1]), tf.reduce_prod(tf.shape(delayIndex)[2:])), [tf.shape(delayIndex)[0]]),
                tf.tile(tf.repeat(tf.range(tf.shape(delayIndex)[2]), tf.shape(delayIndex)[3]*tf.shape(delayIndex)[4]), [tf.shape(delayIndex)[0]*tf.shape(delayIndex)[1]]),
                tf.reshape(delayIndex, [-1])
            ], axis=1)
            print("tf.shape(indices):", tf.shape(indices))


            shape = tf.concat([tf.shape(delayIndex)[:3], [fft_size]], axis=0)
            expanded = tf.zeros(shape, dtype=values.dtype)
            #result = tf.tensor_scatter_nd_update(expanded, indices, tf.reshape(values, [-1]))
            result = tf.tensor_scatter_nd_add(expanded, indices, tf.reshape(values, [-1]))
            return result
        """
        
        def expand_to_delay(values, delayIndex):
            """
            각 요소를 delay index에 맞게 확장하여 최종 텐서로 변환.
            
            Args:
                values: 입력 텐서, shape: [batch_size, N_BS, N_UE, N_cluster, N_rays]
                delayIndex: delay index 텐서, shape: [batch_size, N_BS, N_UE, N_cluster, N_rays]
                fft_size: FFT 크기 (정수), 최종 delay 도메인 차원의 크기
                
            Returns:
                result: 확장된 텐서, shape: [batch_size, N_BS, N_UE, N_cluster*N_rays, fft_size]
                        각 위치에 values의 값이 one-hot 인코딩된 delay index 위치에 배치됨.
            """
            # 1. delayIndex를 one-hot 인코딩: shape -> [B, N_BS, N_UE, N_cluster, N_rays, fft_size]
            one_hot = tf.one_hot(delayIndex, depth=fft_size, dtype=values.dtype)
            
            # 2. values의 마지막 차원을 확장하여 one_hot과 곱할 수 있도록 함: shape -> [B, N_BS, N_UE, N_cluster, N_rays, 1]
            values_expanded = tf.expand_dims(values, axis=-1)
            
            # 3. 각 값이 해당 delay slot에 배치되도록 곱셈 수행
            delay_domain_values = one_hot * values_expanded  # shape: [B, N_BS, N_UE, N_cluster, N_rays, fft_size]
            
            # 4. N_cluster와 N_rays 차원을 하나로 합침
            shape = tf.shape(delay_domain_values)
            result = tf.reshape(delay_domain_values, [shape[0], shape[1], shape[2], shape[3]*shape[4], shape[5]])
            
            return result

        # 각도와 powers를 delay domain으로 확장
        aoa_delay = expand_to_delay(rays.aoa, delayIndex)
        print("tf.shape(aoa_delay):", tf.shape(aoa_delay))

        aod_delay = expand_to_delay(rays.aod, delayIndex)
        print("tf.shape(aod_delay):", tf.shape(aod_delay))

        zoa_delay = expand_to_delay(rays.zoa, delayIndex)
        print("tf.shape(zoa_delay):", tf.shape(zoa_delay))

        zod_delay = expand_to_delay(rays.zod, delayIndex)
        print("tf.shape(zod_delay):", tf.shape(zod_delay))

        xpr_delay = expand_to_delay(rays.xpr, delayIndex)
        print("tf.shape(xpr_delay):", tf.shape(xpr_delay))


        powers_delay = expand_to_delay(rays.powers, delayIndex)
        print("tf.shape(powers_delay):", tf.shape(powers_delay))
        print("========= angles expansion 전체 연산 소요시간:", round((time.time()-tic)*1000,4), "ms")
        print("\n")

        return aoa_delay, aod_delay, zoa_delay, zod_delay, xpr_delay, powers_delay


    def _H_PDP_FIX(self, topology, rays, fft_size, scs):

        # 1. Delay index 계산
        def compute_delay_index():
            delayIndex = tf.cast(tf.floor(rays.delays*scs*fft_size), dtype=tf.int32)
            return delayIndex-tf.reduce_min(delayIndex, axis=-1, keepdims=True)
        
        #delayIndex = measure_cupy(compute_delay_index, 'delay_index', timings)
        delayIndex = compute_delay_index()

        # 2. 각도와 powers를 delay domain으로 확장
        def expand_to_delay(values, delayIndex):
            one_hot = tf.one_hot(delayIndex, depth=fft_size, dtype=values.dtype)
            values_expanded = tf.expand_dims(values, axis=-1)
            delay_domain_values = one_hot * values_expanded
            shape = tf.shape(delay_domain_values)
            return tf.reshape(delay_domain_values, [shape[0], shape[1], shape[2], shape[3]*shape[4], shape[5]])

        def expand_angles():
            aoa_delay = expand_to_delay(rays.aoa, delayIndex)
            aod_delay = expand_to_delay(rays.aod, delayIndex)
            zoa_delay = expand_to_delay(rays.zoa, delayIndex)
            zod_delay = expand_to_delay(rays.zod, delayIndex)
            xpr_delay = expand_to_delay(rays.xpr, delayIndex)
            powers_delay = expand_to_delay(rays.powers, delayIndex)
            return aoa_delay, aod_delay, zoa_delay, zod_delay, xpr_delay, powers_delay

        #aoa_delay, aod_delay, zoa_delay, zod_delay, xpr_delay, powers_delay = measure_cupy(expand_angles, 'angle_expansion', timings)
        aoa_delay, aod_delay, zoa_delay, zod_delay, xpr_delay, powers_delay = expand_angles()
        # 3. Phase matrix 계산
        def compute_phase_matrix():
            phi = self._step_10(tf.shape(aoa_delay))

            # 일단 sqrt(1/xpr_delay) 계산
            raw_scaling = tf.sqrt(1/xpr_delay)
            
            # inf 값을 0으로 대체
            safe_scaling = tf.where(
                tf.math.is_inf(raw_scaling) | tf.math.is_nan(raw_scaling),  # inf나 nan인 경우
                tf.zeros_like(raw_scaling),  # 0으로 대체
                raw_scaling  # 그 외에는 원래 값 유지
            )

            xpr_scaling = tf.complex(safe_scaling, tf.constant(0., self.rdtype))
            e0 = tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,0]))
            e3 = tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,3]))
            e1 = xpr_scaling*tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,1]))
            e2 = xpr_scaling*tf.exp(tf.complex(tf.constant(0., self.rdtype), phi[...,2]))
            shape_phase = tf.concat([tf.shape(e0), [2,2]], axis=-1)
            return tf.reshape(tf.stack([e0, e1, e2, e3], axis=-1), shape_phase)

        #h_phase = measure_cupy(compute_phase_matrix, 'phase_matrix', timings)
        h_phase = compute_phase_matrix()
        #print("h_phase:", h_phase)
        # 4. Field matrix 계산
        def compute_field_matrix():
            tx_orientations = topology.tx_orientations
            rx_orientations = topology.rx_orientations
            
            s = tf.shape(tx_orientations)
            shape = tf.concat([s[:2], [1,1,1,s[-1]]], 0)
            tx_orientations = tf.reshape(tx_orientations, shape)
            
            zod_prime, aod_prime = self._gcs_to_lcs(tx_orientations, zod_delay, aod_delay)
            
            s = tf.shape(rx_orientations)
            shape = tf.concat([[s[0],1],[s[1],1,1,s[-1]]], 0)
            rx_orientations = tf.reshape(rx_orientations, shape)
            
            zoa_prime, aoa_prime = self._gcs_to_lcs(rx_orientations, zoa_delay, aoa_delay)
            
            f_tx_pol1_prime = tf.stack(self._tx_array.ant_pol1.field(zod_prime,aod_prime), axis=-1)
            f_rx_pol1_prime = tf.stack(self._rx_array.ant_pol1.field(zoa_prime,aoa_prime), axis=-1)
            f_tx_pol1 = self._l2g_response(f_tx_pol1_prime, tx_orientations, zod_delay, aod_delay)
            f_rx_pol1 = self._l2g_response(f_rx_pol1_prime, rx_orientations, zoa_delay, aoa_delay)
            
            if self._tx_array.polarization == 'dual':
                f_tx_pol2_prime = tf.stack(self._tx_array.ant_pol2.field(zod_prime, aod_prime), axis=-1)
                f_tx_pol2 = self._l2g_response(f_tx_pol2_prime, tx_orientations, zod_delay, aod_delay)
            
            if self._rx_array.polarization == 'dual':
                f_rx_pol2_prime = tf.stack(self._rx_array.ant_pol2.field(zoa_prime, aoa_prime), axis=-1)
                f_rx_pol2 = self._l2g_response(f_rx_pol2_prime, rx_orientations, zoa_delay, aoa_delay)
            f_tx_pol1_complex = tf.complex(f_tx_pol1, tf.constant(0., self.rdtype))
            pol1_tx = tf.matmul(h_phase, f_tx_pol1_complex)
            """
            print("tf.shape(h_phase):", tf.shape(h_phase))
            print("tf.shape(f_tx_pol1_complex):", tf.shape(f_tx_pol1_complex))
            print("tf.shape(pol1_tx):", tf.shape(pol1_tx))
            print("=========================================")
            print("h_phase[0,0,0,0,0,0,0]:", h_phase[0,0,0,0,0,0,0])
            print("f_tx_pol1[0,0,0,0,0,0,0]:", f_tx_pol1[0,0,0,0,0,0,0])
            print("f_tx_pol1_complex[0,0,0,0,0,0,0]:", f_tx_pol1_complex[0,0,0,0,0,0,0])
            print("pol1_tx[0,0,0,0,0,0,0]:", pol1_tx[0,0,0,0,0,0,0])
            print("=========================================")
            print("\n")
            """
            num_ant_tx = self._tx_array.num_ant
            if self._tx_array.polarization == 'single':
                f_tx_array = tf.tile(tf.expand_dims(pol1_tx, 0),
                    tf.concat([[num_ant_tx], tf.ones([tf.rank(pol1_tx)], tf.int32)], axis=0))
            else:
                pol2_tx = tf.matmul(h_phase, tf.complex(f_tx_pol2, tf.constant(0., self.rdtype)))
                pol_tx = tf.stack([pol1_tx, pol2_tx], 0)
                ant_ind_pol2 = self._tx_array.ant_ind_pol2
                num_ant_pol2 = ant_ind_pol2.shape[0]
                gather_ind = tf.scatter_nd(tf.reshape(ant_ind_pol2, [-1,1]),
                    tf.ones([num_ant_pol2], tf.int32), [num_ant_tx])
                f_tx_array = tf.gather(pol_tx, gather_ind, axis=0)
            num_ant_rx = self._rx_array.num_ant
            if self._rx_array.polarization == 'single':
                f_rx_array = tf.tile(tf.expand_dims(f_rx_pol1, 0),
                    tf.concat([[num_ant_rx], tf.ones([tf.rank(f_rx_pol1)], tf.int32)], axis=0))
                f_rx_array = tf.complex(f_rx_array, tf.constant(0., self.rdtype))
            else:
                pol_rx = tf.stack([f_rx_pol1, f_rx_pol2], 0)
                ant_ind_pol2 = self._rx_array.ant_ind_pol2
                num_ant_pol2 = ant_ind_pol2.shape[0]
                gather_ind = tf.scatter_nd(tf.reshape(ant_ind_pol2, [-1,1]),
                    tf.ones([num_ant_pol2], tf.int32), [num_ant_rx])
                f_rx_array = tf.complex(tf.gather(pol_rx, gather_ind, axis=0),
                    tf.constant(0., self.rdtype))
            h_field = tf.reduce_sum(tf.expand_dims(f_rx_array, 1)*tf.expand_dims(f_tx_array, 0), [-2,-1])
            return tf.transpose(h_field, perm=[2,3,4,5,6,0,1])

        #h_field = measure_cupy(compute_field_matrix, 'field_matrix', timings)
        h_field = compute_field_matrix()
        #print("h_field:", h_field)
        # 5. Array offsets 계산
        def compute_array_offsets():
            lambda_0 = self._lambda_0
            r_hat_rx = self._unit_sphere_vector(zoa_delay, aoa_delay)
            r_hat_rx = tf.squeeze(r_hat_rx, axis=r_hat_rx.shape.rank-1)
            r_hat_tx = self._unit_sphere_vector(zod_delay, aod_delay)
            r_hat_tx = tf.squeeze(r_hat_tx, axis=r_hat_tx.shape.rank-1)
            
            d_bar_rx = self._step_11_get_rx_antenna_positions(topology)
            d_bar_tx = self._step_11_get_tx_antenna_positions(topology)
            r_hat_tx = tf.expand_dims(r_hat_tx, -2)
            r_hat_rx = tf.expand_dims(r_hat_rx, -2)
            
            s = tf.shape(d_bar_tx)
            shape = tf.concat([s[:2], [1,1,1], s[2:]], 0)
            d_bar_tx = tf.reshape(d_bar_tx, shape)
            
            s = tf.shape(d_bar_rx)
            shape = tf.concat([[s[0]], [1,s[1],1,1], s[2:]], 0)
            d_bar_rx = tf.reshape(d_bar_rx, shape)
            
            s = tf.shape(d_bar_rx)
            shape = tf.concat([[s[0]], [tf.shape(r_hat_rx)[1]], s[2:]], 0)
            d_bar_rx = tf.broadcast_to(d_bar_rx, shape)
            
            exp_rx = 2*PI/lambda_0*tf.reduce_sum(r_hat_rx*d_bar_rx, axis=-1, keepdims=True)
            exp_rx = tf.exp(tf.complex(tf.constant(0., self.rdtype), exp_rx))
            
            exp_tx = 2*PI/lambda_0*tf.reduce_sum(r_hat_tx*d_bar_tx, axis=-1)
            exp_tx = tf.exp(tf.complex(tf.constant(0., self.rdtype), exp_tx))
            exp_tx = tf.expand_dims(exp_tx, -2)
            
            return exp_rx*exp_tx

        #h_array = measure_cupy(compute_array_offsets, 'array_offsets', timings)
        h_array = compute_array_offsets()
        #print("h_array[0,0,0,0,0,0,0]:", h_array[0,0,0,0,0,0,0])
        # Doppler matrix 전까지 계산
        def compute_before_doppler():
            h_field_array =  tf.expand_dims(h_field*h_array, -1)
            power_scaling = tf.complex(tf.sqrt(powers_delay), tf.constant(0., self.rdtype))
            power_scaling_reshaped = tf.reshape(power_scaling, tf.concat([tf.shape(power_scaling), [1,1,1]], 0))
            return h_field_array*power_scaling_reshaped
        
        #h_field_array_power = measure_cupy(compute_before_doppler, 'before_doppler', timings)
        h_field_array_power = compute_before_doppler()

        return h_field_array_power, aoa_delay, zoa_delay

    @tf.function(jit_compile=True)
    def _H_TTI(self, topology, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]

        # Doppler matrix 계산

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=-1) #[B, N_UE, 3, 1]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, N_UE, 3, 1]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, N_UE, 1, 3, 1]
        v_bar = tf.expand_dims(v_bar, -3) # [B, 1, 1, N_UE, 3, 1] or [B, 1, N_UE, 1, 3, 1]
        v_bar = tf.expand_dims(v_bar, -3) # 최종적으로 DL [B, 1, 1, 1, N_UE, 3, 1] or UL [B, 1, 1, N_UE, 1, 3, 1]
        
        r_hat_rx = self._unit_sphere_vector(zoa_delay, aoa_delay) #sin, cos 호출
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -2)*sample_times
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -2), -2)


        # Power scaling & 최종 채널 계수 계산       

        h_field_array_power = h_field_array_power_
        h_field_array_power_doppler = h_field_array_power*h_doppler

        mask = tf.expand_dims(ActiveUE, 1) * tf.expand_dims(ServingBS, 0)
        mask = tf.reshape(mask, [1, tf.shape(ServingBS)[0], tf.shape(ActiveUE)[0], 1, 1, 1, 1, 1])

        h_delay_active = h_field_array_power_doppler * mask
        h_delay_active = tf.reduce_sum(h_delay_active, axis=3)
        h_delay_active = tf.transpose(h_delay_active, [0,1,2,4,5,6,3])

        # FFT

        h_freq_serving = tf.signal.fft(h_delay_active)

        # H shape: [B, N_BS, N_UE, N_rxAnt, N_txAnt, N_sym, N_fft]
        
        power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_serving)), axis=[3,4]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함

        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(power, axis=1, keepdims=True)  # shape: [N_Batch, 1, N_UE, N_sym, N_fft], 모든 BS 전력의 합
        interference = total_power - power  # shape: [N_Batch, N_BS, N_UE, N_sym, N_fft] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * power) / (noise_power + tx_power * interference)

        # dl_mimo_SINR_EESM(beta=1.0):

        B      = tf.shape(h_freq_serving)[0]
        N_BS   = tf.shape(h_freq_serving)[1]
        N_UE   = tf.shape(h_freq_serving)[2]
        N_rxAnt   = tf.shape(h_freq_serving)[3]
        N_txAnt   = tf.shape(h_freq_serving)[4]
        N_sym  = tf.shape(h_freq_serving)[5]
        N_fft  = tf.shape(h_freq_serving)[6]
        M = tf.cast(N_sym * N_fft, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS, N_UE, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay_active, h_freq_serving, sinr, sinr_eff
    
    @tf.function(jit_compile=True)
    def _H_TTI_afterDoppler_Precoding(self, topology, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, g_precoding, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_ELW: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW
        # sequential_fft_o_ELW2: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW, 입력 차원 Einsum과 동일 - [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_UE, N_BS, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_UE, N_BS, N_FFT]

        # Doppler matrix 계산

        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx': # Moving_end UE is rx --> Donlink
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        elif topology.moving_end == 'tx': # Moving_end UE is tx--> Uplink
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, N_UE, 1, 1, 3]or UL [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, N_UE, 1, 1, 1, 3] or UL [B, 1, 1, N_UE, 1, 1, 3] 
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_UE, N_BS, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_UE, N_BS, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_UE, N_BS, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_UE, N_BS, N_sym, N_FFT] axis = 2,3 for N_UE_Ant, N_BS_Ant

        # [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler

        # [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay, axis=1) # rays 합산 in frequency domain


        # FFT 내장함수 사용
        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 


        mask = tf.expand_dims(ActiveUE, 0) * tf.expand_dims(ServingBS, 1)
        mask = tf.reshape(mask, [1, 1, 1, tf.shape(ActiveUE)[0], tf.shape(ServingBS)[0], 1, 1])

        # Serving BS, Active UE Masking [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_freq_serving = h_freq*mask # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] * [1, 1, 1, N_UE, N_BS, 1, 1])


        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> [B, N_UE, N_BS, N_sym, N_FFT]
        h_power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_serving)), axis=[1,2]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(h_power, axis=2, keepdims=True)  # shape: [B, N_UE, 1, N_sym, N_FFT], 모든 BS 전력의 합

        
        interference = total_power - h_power  # shape: [B, N_UE, N_BS, N_sym, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)


        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_UE, N_BS, -1])
        
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq_serving, sinr, sinr_eff
    
    #@tf.function(jit_compile=True)
    def _H_TTI_afterDoppler_Precoding_OneHotMasking(self, topology, ActiveUE, ServingBS, g_precoding, g_equalization, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_ELW: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW
        # sequential_fft_o_ELW2: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW, 입력 차원 Einsum과 동일 - [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_UE_Ant, N_BS_Ant, 1]
            # Reshaped: [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, 1, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_UE, N_BS, N_FFT]

        # Doppler matrix 계산

        B, _, _, _, N_UE, N_BS, _, N_FFT = h_field_array_power_.shape
        N_sym = len(sample_times)
        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx': # Moving_end UE is rx --> Donlink
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        elif topology.moving_end == 'tx': # Moving_end UE is tx--> Uplink
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, N_UE, 1, 1, 3]or UL [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, N_UE, 1, 1, 1, 3] or UL [B, 1, 1, N_UE, 1, 1, 3] 
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_UE, N_BS, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_UE, N_BS, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_UE, N_BS, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_UE, N_BS, N_sym, N_FFT] axis = 2,3 for N_UE_Ant, N_BS_Ant

        # [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay_rays = h_field_array_power*h_doppler
        print("h_delay_rays.shape",h_delay_rays.shape)

        # [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay_rays, axis=1) # rays 합산 in frequency domain

        print("h_delay.shape",h_delay.shape)

        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]-> [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS_Serving, N_sym, N_FFT]
        h_delay_active = tf.einsum('abcdefg,di->abciefg', h_delay, ActiveUE)
        h_delay_active_serving = tf.einsum('je, abcdefg->abcdjfg', ServingBS, h_delay_active)


        # FFT 내장함수 사용
        # [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS_Serving, N_sym, N_FFT]
        h_freq_active_serving = tf.signal.fft(h_delay_active_serving) / tf.sqrt(tf.cast(N_FFT,tf.complex64))

        
        ##################################################################
        ################ Precoding & Equalization 수행 필요 ################
        ##################################################################
        # g_precoding shape: [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]

        # [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS_Serving, N_sym, N_FFT] @ [B, N_BS_Ant, N_streams_per_tx, N_BS_Serving, N_sym, N_FFT] 
        # -> [B, N_UE_Ant, N_streams_per_tx, N_UE_Active, N_BS_Serving, N_sym, N_FFT]
        h_precoded = tf.einsum('abcdefg,acxefg->abxdefg', h_freq_active_serving, g_precoding)

        # [B, N_streams_per_UE, N_UE_Ant, N_UE_active, N_sym, N_FFT] @ [B, N_UE_Ant, N_streams_per_tx, N_UE_Active, N_BS_Serving, N_sym, N_FFT]
        # -> [B, N_streams_per_UE, N_streams_per_tx, N_UE_Active, N_BS_Serving, N_sym, N_FFT]
        h_equalized = tf.einsum('awbdfg,abxdefg->awxdefg', g_equalization, h_precoded)
        print("h_equalized.shape",h_equalized.shape)




        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> [B, N_UE, N_BS, N_sym, N_FFT]
        h_power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_active_serving)), axis=[1,2]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(h_power, axis=2, keepdims=True)  # shape: [B, N_UE, 1, N_sym, N_FFT], 모든 BS 전력의 합

        
        interference = total_power - h_power  # shape: [B, N_UE, N_BS, N_sym, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)


        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_UE, N_BS, -1])
        
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay_active_serving, h_freq_active_serving, sinr, sinr_eff
    
    @tf.function(jit_compile=True)
    def _H_TTI_afterDoppler_Precoding_boolean(self, topology, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_ELW: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW
        # sequential_fft_o_ELW2: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW, 입력 차원 Einsum과 동일 - [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_UE, N_BS, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_UE, N_BS, N_FFT]

        # Doppler matrix 계산

        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx': # Moving_end UE is rx --> Donlink
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        elif topology.moving_end == 'tx': # Moving_end UE is tx--> Uplink
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, N_UE, 1, 1, 3]or UL [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, N_UE, 1, 1, 1, 3] or UL [B, 1, 1, N_UE, 1, 1, 3] 
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_UE, N_BS, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_UE, N_BS, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_UE, N_BS, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_UE, N_BS, N_sym, N_FFT] axis = 2,3 for N_UE_Ant, N_BS_Ant

        # [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler

        # [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay, axis=1) # rays 합산 in frequency domain


        # FFT 내장함수 사용
        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 


        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS, N_sym, N_FFT]
        h_freq_active = tf.boolean_mask(h_freq, mask=ActiveUE, axis=3) 
        # [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS, N_sym, N_FFT] -> [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS_Serving, N_sym, N_FFT]
        h_freq_serving_serving = tf.boolean_mask(h_freq_active, mask=ServingBS, axis=4) 

        
        

        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> [B, N_UE, N_BS, N_sym, N_FFT]
        h_power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_serving_serving)), axis=[1,2]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(h_power, axis=2, keepdims=True)  # shape: [B, N_UE, 1, N_sym, N_FFT], 모든 BS 전력의 합

        
        interference = total_power - h_power  # shape: [B, N_UE, N_BS, N_sym, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)


        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_UE, N_BS, -1])
        
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq_serving_serving, sinr, sinr_eff


    # Doppler matrix 계산
    @tf.function(jit_compile=True)
    def _H_TTI_afterDoppler_Precoding_boolean1(self, topology, sample_times, h_field_array_power_, aoa_delay, zoa_delay):

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_UE, N_BS, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_UE, N_BS, N_FFT]

        # Doppler matrix 계산

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx': # Moving_end UE is rx --> Donlink
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        elif topology.moving_end == 'tx': # Moving_end UE is tx--> Uplink
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, N_UE, 1, 1, 3]or UL [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, N_UE, 1, 1, 1, 3] or UL [B, 1, 1, N_UE, 1, 1, 3] 
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_UE, N_BS, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_UE, N_BS, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_UE, N_BS, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_UE, N_BS, N_sym, N_FFT] axis = 2,3 for N_UE_Ant, N_BS_Ant

        # [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler

        # [B, N_Rays, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay, axis=1) # rays 합산 in frequency domain


        # FFT 내장함수 사용
        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 

        return  h_delay, h_freq

    #@tf.function(jit_compile=True)
    #@tf.function()
    def _H_TTI_afterDoppler_Precoding_boolean2(self, ActiveUE, ServingBS, h_freq, snr_dB=10.0, tx_power=1.0, beta=1.0):

        B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT = h_freq.shape

        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS, N_sym, N_FFT]
        h_freq_active = tf.boolean_mask(h_freq, mask=ActiveUE, axis=3) 
        # [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS, N_sym, N_FFT] -> [B, N_UE_Ant, N_BS_Ant, N_UE_Active, N_BS_Serving, N_sym, N_FFT]
        h_freq_serving_serving = tf.boolean_mask(h_freq_active, mask=ServingBS, axis=4) 

        # [B, N_UE_Ant, N_BS_Ant, N_UE, N_BS, N_sym, N_FFT] -> [B, N_UE, N_BS, N_sym, N_FFT]
        h_power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_serving_serving)), axis=[1,2]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(h_power, axis=2, keepdims=True)  # shape: [B, N_UE, 1, N_sym, N_FFT], 모든 BS 전력의 합
        
        interference = total_power - h_power  # shape: [B, N_UE, N_BS, N_sym, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)

        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_UE, N_BS, -1])
        
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  sinr, sinr_eff
    
    @tf.function(jit_compile=True)
    def _H_TTI_oneGraph(self, topology, tau, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        

        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        
        # Doppler matrix 계산

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_FFT, N_BS, N_UE]
        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 3) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, 1, N_UE, 3] or [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # 최종적으로 DL [B, 1, 1, 1, 1, N_UE, 3] or UL [B, 1, 1, 1, N_UE, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified(zoa_delay, aoa_delay) # [B, 1, N_Rays, N_FFT, N_BS, N_UE, 3], axis=1: N_sym

        # [B, 1, N_Rays, N_FFT, N_BS, N_UE] * [1,N_sym,1,1,1,1] = [B, N_sym, N_Rays, N_FFT, N_BS, N_UE]
        sample_len = tf.shape(sample_times)[0]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,sample_len,1,1,1,1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -3), -3) # [B, N_sym, N_Rays, N_FFT, 1, 1, N_BS, N_UE] axis = -3,-4 for N_r, N_t

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler 


        mask = tf.expand_dims(ActiveUE, 1) * tf.expand_dims(ServingBS, 0)
        mask = tf.reshape(mask, [1,1,1,1,1,1, N_BS, N_UE])

        h_delay_active = h_delay*mask


        # FFT 내장함수 사용X, DFT Matrix Element-wise Multiplication 
        # [N_FFT]
        frequencies = self.carrier_frequency * tf.range(-N_FFT/2, N_FFT/2, dtype=self.rdtype) 
        # [1,1,1,N_FFT,1,1,1,1]
        frequencies = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(frequencies, 0), 0), 0), -1), -1), -1), -1) 
        # [B, 1, N_Rays, 1, N_BS, N_UE] -> [B, 1, N_Rays, 1, 1, 1, N_BS, N_UE]
        tau = tf.expand_dims(tf.expand_dims(tau, -3), -3)

        # [B, 1, N_Rays, N_FFT, 1, 1, N_BS, N_UE]
        DFT_exp_factor = tf.exp(tf.complex(tf.constant(0, self.rdtype),-2*PI*frequencies*tau))

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq_active = h_delay_active*DFT_exp_factor

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE] -> # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq_active_raySum = tf.reduce_sum(h_freq_active, axis=2) # rays 합산 in frequency domain

        # [B, N_sym, N_FFT, N_BS, N_UE]
        h_freq_active_power = tf.math.square(tf.reduce_sum(tf.cast(tf.abs(h_freq_active_raySum), tf.float16), axis=[3,4]))
        

        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(h_freq_active_power, axis=-2, keepdims=True)  # shape: [B, N_sym, N_FFT, 1, N_UE], 모든 BS 전력의 합
        interference = total_power - h_freq_active_power  # shape: [B, N_sym, N_FFT, N_BS, N_UE] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_freq_active_power) / (noise_power + tx_power * interference)

        # EESM(beta=1.0):

        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS, N_UE, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay_active, h_freq_active, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_sequential_fft_o_ELW(self, topology, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_ELW: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]

        # Doppler matrix 계산

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=-1) #[B, N_UE, 3, 1]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, N_UE, 3, 1]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, N_UE, 1, 3, 1]
        v_bar = tf.expand_dims(v_bar, -3) # [B, 1, 1, N_UE, 3, 1] or [B, 1, N_UE, 1, 3, 1]
        v_bar = tf.expand_dims(v_bar, -3) # 최종적으로 DL [B, 1, 1, 1, N_UE, 3, 1] or UL [B, 1, 1, N_UE, 1, 3, 1]
        
        r_hat_rx = self._unit_sphere_vector(zoa_delay, aoa_delay) #sin, cos 호출
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -2)*sample_times
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -2), -2)


        # Power scaling & 최종 채널 계수 계산       

        h_field_array_power = h_field_array_power_
        h_field_array_power_doppler = h_field_array_power*h_doppler

        # [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym] -> [B, N_BS, N_UE, N_FFT, N_r, N_t, N_sym]
        h_delay = tf.reduce_sum(h_field_array_power_doppler, axis=3)
        # [B, N_BS, N_UE, N_FFT, N_r, N_t, N_sym] -> [B, N_BS, N_UE, N_rxAnt, N_txAnt, N_sym, N_FFT]
        h_delay = tf.transpose(h_delay, [0,1,2,4,5,6,3])

        ####################################################################################################################################
        #  fft.84.0
        ####################################################################################################################################
        # [B, N_BS, N_UE, N_rxAnt, N_txAnt, N_sym, N_FFT]
        h_freq = tf.signal.fft(h_delay)


        ####################################################################################################################################
        #  loop_multiply_fusion.2 
        ####################################################################################################################################

        mask = tf.expand_dims(ActiveUE, 1) * tf.expand_dims(ServingBS, 0)
        mask = tf.reshape(mask, [1, tf.shape(ServingBS)[0], tf.shape(ActiveUE)[0], 1, 1, 1, 1])

        # Serving BS, Active UE Masking
        h_freq_serving = h_freq*mask # [B, N_BS, N_UE, N_rxAnt, N_txAnt, N_sym, N_FFT] * [1, N_BS, N_UE, 1, 1, 1, 1, 1])
        

        ####################################################################################################################################
        #  input_divide_reduce_fusion.1 
        ####################################################################################################################################

        # [B, ServingBS, ActiveUE, N_rxAnt, N_txAnt, N_sym, N_FFT] -> [B, N_BS, N_UE, N_sym, N_FFT]
        h_power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_serving)), axis=[3,4]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        ##################################################################
        #  input_reduce_fusion.2 
        ##################################################################
        total_power = tf.reduce_sum(h_power, axis=1, keepdims=True)  # shape: [B, 1, N_UE, N_sym, N_FFT], 모든 BS 전력의 합

        
        interference = total_power - h_power  # shape: [B, N_BS, N_UE, N_sym, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)

        # dl_mimo_SINR_EESM(beta=1.0):

        B      = tf.shape(h_freq_serving)[0]
        N_BS   = tf.shape(h_freq_serving)[1]
        N_UE   = tf.shape(h_freq_serving)[2]
        N_rxAnt   = tf.shape(h_freq_serving)[3]
        N_txAnt   = tf.shape(h_freq_serving)[4]
        N_sym  = tf.shape(h_freq_serving)[5]
        N_fft  = tf.shape(h_freq_serving)[6]
        M = tf.cast(N_sym * N_fft, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS, N_UE, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq_serving, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_sequential_fft_o_ELW2_noProfile(self, topology, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_ELW: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW
        # sequential_fft_o_ELW2: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW, 입력 차원 Einsum과 동일 - [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_BS, N_UE, N_FFT]

        # Doppler matrix 계산

        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, 1, N_UE, 1, 3] or UL [B, 1, N_UE, 1, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, 1, N_UE, 1, 1, 3] or UL [B, 1, N_UE, 1, 1, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym
        # [B, N_Rays, N_BS, N_UE, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_BS, N_UE, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT] axis = 2,3 for N_r, N_t
        #print("tf.reduce_mean(h_doppler):", tf.reduce_mean(h_doppler))

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay, axis=1) # rays 합산 in frequency domain


        # FFT 내장함수 사용
        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 


        mask = tf.expand_dims(ActiveUE, 0) * tf.expand_dims(ServingBS, 1)
        mask = tf.reshape(mask, [1, 1, 1, tf.shape(ServingBS)[0], tf.shape(ActiveUE)[0], 1, 1])

        # Serving BS, Active UE Masking [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq_serving = h_freq*mask # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] * [1, 1, 1, N_BS, N_UE, 1, 1])


        # [B, N_r, N_t, ServingBS, ActiveUE, N_sym, N_FFT] -> [B, ServingBS, ActiveUE, N_sym, N_FFT]
        h_power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_serving)), axis=[1,2]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(h_power, axis=1, keepdims=True)  # shape: [B, 1, ActiveUE, N_sym, N_FFT], 모든 BS 전력의 합

        
        interference = total_power - h_power  # shape: [B, ServingBS, ActiveUE, N_sym, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)


        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS, N_UE, -1])
        
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq_serving, sinr, sinr_eff
    
    def _H_TTI_sequential_fft_o_ELW2_Profile(self, topology, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_ELW: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW
        # sequential_fft_o_ELW2: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW, 입력 차원 Einsum과 동일 - [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_BS, N_UE, N_FFT]

        # Doppler matrix 계산

        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, 1, N_UE, 1, 3] or UL [B, 1, N_UE, 1, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, 1, N_UE, 1, 1, 3] or UL [B, 1, N_UE, 1, 1, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_BS, N_UE, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_BS, N_UE, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT] axis = 2,3 for N_r, N_t

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay, axis=1) # rays 합산 in frequency domain


        # FFT 내장함수 사용
        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 


        mask = tf.expand_dims(ActiveUE, 0) * tf.expand_dims(ServingBS, 1)
        mask = tf.reshape(mask, [1, 1, 1, tf.shape(ServingBS)[0], tf.shape(ActiveUE)[0], 1, 1])

        # Serving BS, Active UE Masking [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq_serving = h_freq*mask # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] * [1, 1, 1, N_BS, N_UE, 1, 1])


        # [B, N_r, N_t, ServingBS, ActiveUE, N_sym, N_FFT] -> [B, ServingBS, ActiveUE, N_sym, N_FFT]
        h_power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_serving)), axis=[1,2]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(h_power, axis=1, keepdims=True)  # shape: [B, 1, ActiveUE, N_sym, N_FFT], 모든 BS 전력의 합

        
        interference = total_power - h_power  # shape: [B, ServingBS, ActiveUE, N_sym, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)


        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS, N_UE, -1])
        
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq_serving, sinr, sinr_eff
    
    @tf.function(jit_compile=True)
    def _H_TTI_sequential_fft_o_ELW3(self, topology, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_ELW: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW
        # sequential_fft_o_ELW2: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW, 입력 차원 Einsum과 동일 - [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # sequential_fft_o_ELW3: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW, 입력 차원 Einsum과 동일, tf.booelan_mask 사용

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_BS, N_UE, N_FFT]

        # Doppler matrix 계산

        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, 1, N_UE, 1, 3] or UL [B, 1, N_UE, 1, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, 1, N_UE, 1, 1, 3] or UL [B, 1, N_UE, 1, 1, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_BS, N_UE, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_BS, N_UE, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT] axis = 2,3 for N_r, N_t

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay, axis=1) # rays 합산 in frequency domain

        # [B, N_r, N_t, N_BS, N_UE_Active, N_sym, N_FFT]
        h_delay_active = tf.boolean_mask(h_delay, mask=ActiveUE, axis=4) 
        # [B, N_r, N_t, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        h_delay_serving = tf.boolean_mask(h_delay_active, mask=ServingBS, axis=3) 


        # FFT 내장함수 사용
        # [B, N_r, N_t, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq_serving = tf.signal.fft(h_delay_serving) 

        

        # [B, N_r, N_t, ServingBS, ActiveUE, N_sym, N_FFT] -> [B, ServingBS, ActiveUE, N_sym, N_FFT]
        h_power = tf.reduce_sum(tf.math.square(tf.abs(h_freq_serving)), axis=[1,2]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(h_power, axis=1, keepdims=True)  # shape: [B, 1, ActiveUE, N_sym, N_FFT], 모든 BS 전력의 합

        
        interference = total_power - h_power  # shape: [B, ServingBS, ActiveUE, N_sym, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)

        # dl_mimo_SINR_EESM(beta=1.0):

        N_BS_Serving = tf.shape(ServingBS)[0]
        N_UE_Active = tf.shape(ActiveUE)[0]

        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq_serving, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_sequential_fft_o_Einsum(self, topology, mask_UE, mask_BS, ones_BS_Serving, ones_UE_Active, eye_BS_interference, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_matmul_x: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW
        # sequential_fft_o_matmul_o: FFT 내장함수 사용, BS,UE Masking + SINR 계산 matmul

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_BS, N_UE, N_FFT]

        # Doppler matrix 계산

        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, 1, N_UE, 1, 3] or UL [B, 1, N_UE, 1, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, 1, N_UE, 1, 1, 3] or UL [B, 1, N_UE, 1, 1, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_BS, N_UE, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_BS, N_UE, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT] axis = 2,3 for N_r, N_t

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay, axis=1) # rays 합산 in frequency domain


        # FFT 내장함수 사용
        #[B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 

        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq_abs = tf.cast(tf.abs(h_freq), tf.float16)

        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> [B, N_BS, N_UE, N_sym, N_FFT]
        # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        h_freq_abs = tf.reduce_sum(h_freq_abs, axis=[1,2])

        # [B, N_BS, N_UE, N_sym, N_FFT] -> [B, N_BS, N_UE_Active, N_sym, N_FFT]
        h_abs_active = tf.einsum('bmnqw,nk->bmkqw', h_freq_abs, mask_UE) 
        h_abs_serving_active = tf.einsum('lm,bmkqw->blkqw', mask_BS, h_abs_active)   
        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        h_power = tf.math.square(h_abs_serving_active) 

        # ones_BS_Serving: [N_BS_Serving, 1]
        # ones_UE_Active:  [1, N_UE_Active]
        # total_power = tf.matmul(ones_BS_Serving, h_power)  # [B*N_sym*N_FFT, 1, N_UE_Active]
        
        # Matmul 사용한 interference 계산
        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        #interference = tf.matmul(eye_BS_interference, h_power)
        interference = tf.einsum('xb,abcde->axcde', eye_BS_interference, h_power)

        #interference = total_power - h_power  # same shape

        # tx_power, snr_dB 주어졌다고 가정
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin

        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)


        N_BS_Serving = ones_BS_Serving.shape[1] # [1,N_BS_serving]
        N_UE_Active = ones_UE_Active.shape[0] # [N_UE_active, 1]
        
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_sequential_fft_o_Einsum2(self, topology, mask_UE, mask_BS, ones_BS_Serving, ones_UE_Active, eye_BS_interference, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 
        # oneGraph: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 모두 ELW
        # sequential_fft_o_ELW: FFT 내장함수 사용, BS,UE Masking + SINR 계산 ELW
        # sequential_fft_o_Einsum: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum
        # sequential_fft_o_Einsum2: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum2, ELW와 입력 차원 동일

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_BS, N_UE, N_FFT]

        # Doppler matrix 계산

        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, 1, N_UE, 1, 3] or UL [B, 1, N_UE, 1, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, 1, N_UE, 1, 1, 3] or UL [B, 1, N_UE, 1, 1, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_BS, N_UE, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_BS, N_UE, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT] axis = 2,3 for N_r, N_t

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_delay = tf.reduce_sum(h_delay, axis=1) # rays 합산 in frequency domain


        # FFT 내장함수 사용
        #[B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 

        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq_abs = tf.cast(tf.abs(h_freq), tf.float16)

        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> [B, N_BS, N_UE, N_sym, N_FFT]
        # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        h_freq_abs = tf.reduce_sum(h_freq_abs, axis=[1,2])

        # [B, N_BS, N_UE, N_sym, N_FFT] -> [B, N_BS, N_UE_Active, N_sym, N_FFT]
        h_abs_active = tf.einsum('bmnqw,nk->bmkqw', h_freq_abs, mask_UE) 
        h_abs_serving_active = tf.einsum('lm,bmkqw->blkqw', mask_BS, h_abs_active)   
        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        h_power = tf.math.square(h_abs_serving_active) 

        # ones_BS_Serving: [N_BS_Serving, 1]
        # ones_UE_Active:  [1, N_UE_Active]
        # total_power = tf.matmul(ones_BS_Serving, h_power)  # [B*N_sym*N_FFT, 1, N_UE_Active]
        
        # Matmul 사용한 interference 계산
        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        #interference = tf.matmul(eye_BS_interference, h_power)
        interference = tf.einsum('xb,abcde->axcde', eye_BS_interference, h_power)

        #interference = total_power - h_power  # same shape

        # tx_power, snr_dB 주어졌다고 가정
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin

        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)


        N_BS_Serving = ones_BS_Serving.shape[1] # [1,N_BS_serving]
        N_UE_Active = ones_UE_Active.shape[0] # [N_UE_active, 1]
        
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_TC1(self, topology, mask_UE, mask_BS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
    
        # Doppler matrix 계산
        """
        # h_field_array_power_ shape
            # Original: [B, BS, UE, Rays, FFT, N_r, N_t, N_sym]
            # Reshaped: [B, Rays, FFT, N_r, N_t, N_sym, BS, UE]
        # zoa_delay, aoa_delay shape
            # Original: [B, BS, UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_FFT, BS, UE]
        
        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=-1) #[B, N_UE, 3, 1]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, N_UE, 3, 1]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, N_UE, 1, 3, 1]
        v_bar = tf.expand_dims(v_bar, 2) # [B, 1, 1, N_UE, 3, 1] or [B, 1, N_UE, 1, 3, 1]
        v_bar = tf.expand_dims(v_bar, 2) # 최종적으로 DL [B, 1, 1, 1, N_UE, 3, 1] or UL [B, 1, 1, N_UE, 1, 3, 1]
        r_hat_rx = self._unit_sphere_vector(zoa_delay, aoa_delay) # [B, N_Rays, N_FFT, N_BS, N_UE, 3, 1]
        """

        # h_field_array_power_ shape
            # Original: [B, BS, UE, Rays, FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_sym, Rays, FFT, N_r, N_t, BS, UE]
        # zoa_delay, aoa_delay shape
            # Original: [B, BS, UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_FFT, BS, UE]

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 3) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, 1, N_UE, 3] or [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # 최종적으로 DL [B, 1, 1, 1, 1, N_UE, 3] or UL [B, 1, 1, 1, N_UE, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified(zoa_delay, aoa_delay) # [B, 1, N_Rays, N_FFT, N_BS, N_UE, 3], axis=1: N_sym


        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,len(sample_times),1,1,1,1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -4), -4) # [B, N_Rays, N_FFT, 1, 1, N_sym, N_BS, N_UE] 1,1 for N_r, N_t

        # [B, N_Rays, N_FFT, N_r, N_t, N_sym, N_BS, N_UE]
        h_field_array_power = h_field_array_power_
        h_field_array_power_doppler = h_field_array_power*h_doppler 

        # [B, N_Rays, N_FFT, N_r, N_t, N_sym, N_BS, N_UE] -> # [B, N_FFT, N_r, N_t, N_sym, N_BS, N_UE]
        h_delay = tf.reduce_sum(h_field_array_power_doppler, axis=1)


        # [B, N_FFT, N_r, N_t, N_sym, N_BS, N_UE] -> [B, N_r, N_t, N_sym, N_BS, N_UE, N_FFT]
        h_delay = tf.transpose(h_delay, [0,2,3,4,5,6,1])

        # FFT

        # [B, N_r, N_t, N_sym, N_BS, N_UE, N_FFT]
        h_freq = tf.signal.fft(h_delay)
        
        h_freq_abs = tf.cast(tf.abs(h_freq), tf.float16)
        
        # [B, N_r, N_t, N_sym, N_BS, N_UE, N_FFT] -> [B, N_r, N_t, N_sym, N_FFT, N_BS, N_UE]
        h_freq_abs = tf.transpose(h_freq_abs, [0,1,2,3,6,4,5])

        # [B, N_r, N_t, N_sym, N_FFT, N_BS, N_UE] -> [B, N_r, N_t, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        h_freq_abs_serving = tf.matmul(mask_BS, tf.matmul(h_freq_abs, mask_UE))

        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        power = tf.reduce_sum(tf.math.square(h_freq_abs), axis=[1,2]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함

        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        total_power = tf.reduce_sum(power, axis=2, keepdims=True)  # shape: [B, N_sym, 1, N_BS_Serving, N_UE_Active, N_FFT], 모든 BS 전력의 합
        interference = total_power - power  # [B, N_sym, N_BS_Serving, N_UE_Active, N_FFT] total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        sinr = (tx_power * power) / (noise_power + tx_power * interference) # [B, N_sym, N_BS_Serving, N_UE_Active, N_FFT]

        # h_freq_serving: [B, N_r, N_t, N_sym, N_BS_Serving, N_UE_Active, N_FFT] 
        B            = tf.shape(h_freq_abs_serving)[0]
        N_r          = tf.shape(h_freq_abs_serving)[1]
        N_t          = tf.shape(h_freq_abs_serving)[2]
        N_sym        = tf.shape(h_freq_abs_serving)[3]
        N_BS_Serving = tf.shape(h_freq_abs_serving)[4]
        N_UE_Active  = tf.shape(h_freq_abs_serving)[5]
        N_FFT        = tf.shape(h_freq_abs_serving)[6]
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq, sinr, sinr_eff
            
    @tf.function(jit_compile=True)
    def _H_TTI_TC2(self, topology, tau, mask_UE, mask_BS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        
        # Doppler matrix 계산

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_FFT, N_BS, N_UE]
        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 3) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, 1, N_UE, 3] or [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # 최종적으로 DL [B, 1, 1, 1, 1, N_UE, 3] or UL [B, 1, 1, 1, N_UE, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified(zoa_delay, aoa_delay) # [B, 1, N_Rays, N_FFT, N_BS, N_UE, 3], axis=1: N_sym

        # [B, 1, N_Rays, N_FFT, N_BS, N_UE] * [1,N_sym,1,1,1,1] = [B, N_sym, N_Rays, N_FFT, N_BS, N_UE]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,len(sample_times),1,1,1,1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -3), -3) # [B, N_sym, N_Rays, N_FFT, 1, 1, N_BS, N_UE] axis = -3,-4 for N_r, N_t

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler 


        # FFT X, DFT Matrix Element-wise Multiplication

        frequencies = self.carrier_frequency * tf.range(-N_FFT/2, N_FFT/2, dtype=self.rdtype) # [N_FFT]

        # [1,1,1,N_FFT,1,1,1,1]
        frequencies = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(frequencies, 0), 0), 0), -1), -1), -1), -1) 
        # [B, 1, N_Rays, 1, N_BS, N_UE] -> [B, 1, N_Rays, 1, 1, 1, N_BS, N_UE]
        tau = tf.expand_dims(tf.expand_dims(tau, -3), -3)

        # [B, 1, N_Rays, N_FFT, 1, 1, N_BS, N_UE]
        DFT_exp_factor = tf.exp(tf.complex(tf.constant(0, self.rdtype),-2*PI*frequencies*tau))

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = h_delay*DFT_exp_factor

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE] -> # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = tf.reduce_sum(h_freq, axis=2) # rays 합산 in frequency domain

        # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq_abs = tf.cast(tf.abs(h_freq), tf.float16)

        # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE] -> [B, N_sym, N_FFT, N_r, N_t, N_BS_Serving, N_UE_Active]
        h_freq_abs_serving = tf.matmul(mask_BS, tf.matmul(h_freq_abs, mask_UE))

        # [B, N_sym, N_FFT, N_r, N_t, N_BS_Serving, N_UE_Active] ->[B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        power = tf.reduce_sum(tf.math.square(h_freq_abs_serving), axis=[3,4]) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함

        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin  # 예: tx_power=1이면 noise_power = 0.1

        # [B, N_sym, N_FFT, 1, N_UE_Active], 모든 BS 전력의 합
        total_power = tf.reduce_sum(power, axis=-2, keepdims=True)  
        interference = total_power - power  # # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active], 모든 BS 전력의 합 total_power에서 각 BS의 전력을 빼서 간섭 전력을 계산

        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        sinr = (tx_power * power) / (noise_power + tx_power * interference) 

        # h_freq_serving: [B, N_r, N_t, N_sym, N_BS_Serving, N_UE_Active, N_FFT] 
        N_BS_Serving = h_freq_abs_serving.shape[4]
        N_UE_Active = h_freq_abs_serving.shape[5]
       
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq, sinr, sinr_eff


    @tf.function(jit_compile=True)
    def _H_TTI_TC3(self, topology, tau, mask_UE, mask_BS, ones_BS_Serving, ones_UE_Active, eye_BS_interference, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):

        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용

        # Doppler matrix 계산

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_FFT, N_BS, N_UE]
        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 3) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, 1, N_UE, 3] or [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # 최종적으로 DL [B, 1, 1, 1, 1, N_UE, 3] or UL [B, 1, 1, 1, N_UE, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified(zoa_delay, aoa_delay) # [B, 1, N_Rays, N_FFT, N_BS, N_UE, 3], axis=1: N_sym

        # [B, 1, N_Rays, N_FFT, N_BS, N_UE] * [1,N_sym,1,1,1,1] = [B, N_sym, N_Rays, N_FFT, N_BS, N_UE]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,len(sample_times),1,1,1,1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -3), -3) # [B, N_sym, N_Rays, N_FFT, 1, 1, N_BS, N_UE] axis = -3,-4 for N_r, N_t

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler 

        # FFT 내장함수 사용X, DFT Matrix Element-wise Multiplication 
        # [N_FFT]
        frequencies = self.carrier_frequency * tf.range(-N_FFT/2, N_FFT/2, dtype=self.rdtype) 
        # [1,1,1,N_FFT,1,1,1,1]
        frequencies = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(frequencies, 0), 0), 0), -1), -1), -1), -1) 
        # [B, 1, N_Rays, 1, N_BS, N_UE] -> [B, 1, N_Rays, 1, 1, 1, N_BS, N_UE]
        tau = tf.expand_dims(tf.expand_dims(tau, -3), -3)

        # [B, 1, N_Rays, N_FFT, 1, 1, N_BS, N_UE]
        DFT_exp_factor = tf.exp(tf.complex(tf.constant(0, self.rdtype),-2*PI*frequencies*tau))

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = h_delay*DFT_exp_factor

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE] -> # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = tf.reduce_sum(h_freq, axis=2) # rays 합산 in frequency domain

        # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq_abs = tf.cast(tf.abs(h_freq), tf.float16)

        # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE] -> [B, N_sym, N_FFT, N_BS, N_UE]
        h_freq_abs = tf.reduce_sum(h_freq_abs, axis=[3,4])

        # [B, N_sym, N_FFT, N_BS, N_UE] -> [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        h_freq_abs_serving = tf.matmul(mask_BS, tf.matmul(h_freq_abs, mask_UE)) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함

        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        h_power = tf.math.square(h_freq_abs_serving) 

        # ones_BS_Serving: [N_BS_Serving, 1]
        # ones_UE_Active:  [1, N_UE_Active]
        # total_power = tf.matmul(ones_BS_Serving, h_power)  # [B*N_sym*N_FFT, 1, N_UE_Active]
        
        # Matmul 사용한 interference 계산
        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        interference = tf.matmul(eye_BS_interference, h_power)

        #interference = total_power - h_power  # same shape

        # tx_power, snr_dB 주어졌다고 가정
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin

        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)

        # h_freq_serving: [B, N_r, N_t, N_sym, N_BS_Serving, N_UE_Active, N_FFT] 
        N_BS_Serving = ones_BS_Serving.shape[-2]
        N_UE_Active = ones_UE_Active.shape[-1]
        
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq, sinr, sinr_eff
    
    @tf.function(jit_compile=True)
    def _H_TTI_TC4(self, topology, tau, mask_UE, mask_BS, ones_BS_Serving, ones_UE_Active, eye_BS_interference, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):

        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # Doppler matrix 계산

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_FFT, N_BS, N_UE]
        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 3) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, 1, N_UE, 3] or [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # 최종적으로 DL [B, 1, 1, 1, 1, N_UE, 3] or UL [B, 1, 1, 1, N_UE, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified(zoa_delay, aoa_delay) # [B, 1, N_Rays, N_FFT, N_BS, N_UE, 3], axis=1: N_sym

        # [B, 1, N_Rays, N_FFT, N_BS, N_UE] * [1,N_sym,1,1,1,1] = [B, N_sym, N_Rays, N_FFT, N_BS, N_UE]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,len(sample_times),1,1,1,1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -3), -3) # [B, N_sym, N_Rays, N_FFT, 1, 1, N_BS, N_UE] axis = -3,-4 for N_r, N_t

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler 

        # FFT 내장함수 사용X, DFT Matrix Element-wise Multiplication 
        # [N_FFT]
        frequencies = self.carrier_frequency * tf.range(-N_FFT/2, N_FFT/2, dtype=self.rdtype) 
        # [1,1,1,N_FFT,1,1,1,1]
        frequencies = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(frequencies, 0), 0), 0), -1), -1), -1), -1) 
        # [B, 1, N_Rays, 1, N_BS, N_UE] -> [B, 1, N_Rays, 1, 1, 1, N_BS, N_UE]
        tau = tf.expand_dims(tf.expand_dims(tau, -3), -3)

        # [B, 1, N_Rays, N_FFT, 1, 1, N_BS, N_UE]
        DFT_exp_factor = tf.exp(tf.complex(tf.constant(0, self.rdtype),-2*PI*frequencies*tau))

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = h_delay*DFT_exp_factor

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE] -> # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = tf.reduce_sum(h_freq, axis=2) # rays 합산 in frequency domain

        # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq_abs = tf.cast(tf.abs(h_freq), tf.float16)

        # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE] -> [B * N_sym * N_FFT, N_BS, N_UE]
        h_freq_abs = tf.reshape(tf.reduce_sum(h_freq_abs, axis=[3,4]), [B * N_sym * N_FFT, N_BS, N_UE])

        # [B * N_sym * N_FFT, N_BS, N_UE] -> [B * N_sym * N_FFT, N_BS_Serving, N_UE_Active]
        h_freq_abs_serving = tf.matmul(mask_BS, tf.matmul(h_freq_abs, mask_UE)) # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함

        # [B * N_sym * N_FFT, N_BS_Serving, N_UE_Active]
        h_power = tf.math.square(h_freq_abs_serving) 

        # ones_BS_Serving: [N_BS_Serving, 1]
        # ones_UE_Active:  [1, N_UE_Active]
        # total_power = tf.matmul(ones_BS_Serving, h_power)  # [B*N_sym*N_FFT, 1, N_UE_Active]
        
        # Matmul 사용한 interference 계산
        # [B * N_sym * N_FFT, N_BS_Serving, N_UE_Active]
        interference = tf.matmul(eye_BS_interference, h_power)

        #interference = total_power - h_power  # same shape

        # tx_power, snr_dB 주어졌다고 가정
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin

        # # [B * N_sym * N_FFT, N_BS_Serving, N_UE_Active]
        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)

        # h_freq_serving: [B, N_r, N_t, N_sym, N_BS_Serving, N_UE_Active, N_FFT] 
        N_BS_Serving = ones_BS_Serving.shape[0]
        N_UE_Active = ones_UE_Active.shape[1]
        
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, -1, N_BS_Serving, N_UE_Active])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_TC5(self, topology, tau, mask_UE, mask_BS, ones_BS_Serving, ones_UE_Active, eye_BS_interference, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):

        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # Doppler matrix 계산

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_FFT, N_BS, N_UE]
        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 3) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, 1, N_UE, 3] or [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # 최종적으로 DL [B, 1, 1, 1, 1, N_UE, 3] or UL [B, 1, 1, 1, N_UE, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified(zoa_delay, aoa_delay) # [B, 1, N_Rays, N_FFT, N_BS, N_UE, 3], axis=1: N_sym

        # [B, 1, N_Rays, N_FFT, N_BS, N_UE] * [1,N_sym,1,1,1,1] = [B, N_sym, N_Rays, N_FFT, N_BS, N_UE]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,len(sample_times),1,1,1,1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -3), -3) # [B, N_sym, N_Rays, N_FFT, 1, 1, N_BS, N_UE] axis = -3,-4 for N_r, N_t

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler 

        # FFT 내장함수 사용X, DFT Matrix Element-wise Multiplication 
        # [N_FFT]
        frequencies = self.carrier_frequency * tf.range(-N_FFT/2, N_FFT/2, dtype=self.rdtype) 
        # [1,1,1,N_FFT,1,1,1,1]
        frequencies = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(frequencies, 0), 0), 0), -1), -1), -1), -1) 
        # [B, 1, N_Rays, 1, N_BS, N_UE] -> [B, 1, N_Rays, 1, 1, 1, N_BS, N_UE]
        tau = tf.expand_dims(tf.expand_dims(tau, -3), -3)

        # [B, 1, N_Rays, N_FFT, 1, 1, N_BS, N_UE]
        DFT_exp_factor = tf.exp(tf.complex(tf.constant(0, self.rdtype),-2*PI*frequencies*tau))

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = h_delay*DFT_exp_factor

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE] -> # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = tf.reduce_sum(h_freq, axis=2) # rays 합산 in frequency domain

        # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq_abs = tf.cast(tf.abs(h_freq), tf.float16)

        # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE] -> [B, N_sym, N_FFT, N_BS, N_UE]
        h_freq_abs = tf.reduce_sum(h_freq_abs, axis=[3,4])

        # [B, N_sym, N_FFT, N_BS, N_UE] -> [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        #h_freq_abs_serving = tf.matmul(mask_BS, tf.matmul(h_freq_abs, mask_UE)) 
        h_serv = tf.einsum('bsfij,jk->bsfik', h_freq_abs, mask_UE) 
        h_freq_abs_serving = tf.einsum('li,bsfik->bsflk', mask_BS, h_serv)   
        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        h_power = tf.math.square(h_freq_abs_serving) 

        # ones_BS_Serving: [N_BS_Serving, 1]
        # ones_UE_Active:  [1, N_UE_Active]
        # total_power = tf.matmul(ones_BS_Serving, h_power)  # [B*N_sym*N_FFT, 1, N_UE_Active]
        
        # Matmul 사용한 interference 계산
        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        interference = tf.matmul(eye_BS_interference, h_power)

        #interference = total_power - h_power  # same shape

        # tx_power, snr_dB 주어졌다고 가정
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin

        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)

        # h_freq_serving: [B, N_r, N_t, N_sym, N_BS_Serving, N_UE_Active, N_FFT] 
        N_BS_Serving = ones_BS_Serving.shape[-2]
        N_UE_Active = ones_UE_Active.shape[-1]
        
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_TC6(self, topology, tau, mask_UE, mask_BS, ones_BS_Serving, ones_UE_Active, eye_BS_interference, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):

        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # Doppler matrix 계산

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_BS, N_UE, N_FFT]
        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, 1, N_UE, 1, 3] or UL [B, 1, N_UE, 1, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, 1, N_UE, 1, 1, 3] or UL [B, 1, N_UE, 1, 1, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_BS, N_UE, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_BS, N_UE, N_sym, N_FFT]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,1,1,1,len(sample_times),1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT] axis = 2,3 for N_r, N_t

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler 

        # FFT 내장함수 사용
        #[B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq = tf.reduce_sum(h_freq, axis=1) # rays 합산 in frequency domain

        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq_abs = tf.cast(tf.abs(h_freq), tf.float16)

        # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> [B, N_BS, N_UE, N_sym, N_FFT]
        # 여기서 per_stream or Not 나눠서 txAnt 차원을 steram 차원으로 할지 안할지 결정해야 함
        h_freq_abs = tf.reduce_sum(h_freq_abs, axis=[1,2])

        # [B, N_BS, N_UE, N_sym, N_FFT] -> [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        h_serv = tf.einsum('bmnqw,nk->bmkqw', h_freq_abs, mask_UE) 
        h_freq_abs_serving = tf.einsum('lm,bmkqw->blkqw', mask_BS, h_serv)   
        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        h_power = tf.math.square(h_freq_abs_serving) 

        # ones_BS_Serving: [N_BS_Serving, 1]
        # ones_UE_Active:  [1, N_UE_Active]
        # total_power = tf.matmul(ones_BS_Serving, h_power)  # [B*N_sym*N_FFT, 1, N_UE_Active]
        
        # Matmul 사용한 interference 계산
        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        #interference = tf.matmul(eye_BS_interference, h_power)
        interference = tf.einsum('xb,abcde->axcde', eye_BS_interference, h_power)

        #interference = total_power - h_power  # same shape

        # tx_power, snr_dB 주어졌다고 가정
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin

        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        sinr = (tx_power * h_power) / (noise_power + tx_power * interference)


        N_BS_Serving = ones_BS_Serving.shape[1] # [1,N_BS_serving]
        N_UE_Active = ones_UE_Active.shape[0] # [N_UE_active, 1]
        
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)  # log(sum(exp(-sinr_flat/beta)))
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))

        return  h_delay, h_freq, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_TC7(self, topology, tau, mask_UE, mask_BS, eye_BS_interference, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):

        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 총합
        # Doppler matrix 계산

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_BS, N_UE, N_FFT]
        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, -2) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # DL [B, 1, 1, N_UE, 1, 3] or UL [B, 1, N_UE, 1, 1, 3]
        v_bar = tf.expand_dims(v_bar, -2) # 최종적으로 DL [B, 1, 1, N_UE, 1, 1, 3] or UL [B, 1, N_UE, 1, 1, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified2(zoa_delay, aoa_delay) # [B, N_Rays, N_BS, N_UE, 1, N_FFT, 3], axis=-3 -> N_sym

        # [B, N_Rays, N_BS, N_UE, 1, N_FFT] * [1,1,1,1,N_sym,1] = [B, N_Rays, N_BS, N_UE, N_sym, N_FFT]
        sample_len = tf.shape(sample_times)[0]
        exponent = 2 * PI / self._lambda_0 * tf.reduce_sum(r_hat_rx * v_bar, axis=-1) * tf.reshape(sample_times, [1,1,1,1,sample_len,1])

        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, 2), 2) # [B, N_Rays, 1, 1, N_BS, N_UE, N_sym, N_FFT] axis = 2,3 for N_r, N_t

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler 

        # FFT 내장함수 사용
        #[B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        #h_freq = tf.signal.fft(h_delay) / tf.cast(tf.sqrt(N_FFT), tf.complex64)
        h_freq = tf.signal.fft(h_delay) 

        # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT] -> # [B, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
        h_freq = tf.reduce_sum(h_freq, axis=1) # rays 합산 in frequency domain




        # [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]
        # Inline된 einsum + masking + square + interference + SINR 계산
        sinr = tf.math.divide_no_nan(
            tx_power * tf.math.square(
                tf.einsum(
                    'lm,bmkqw->blkqw',
                    mask_BS,
                    tf.einsum(
                        'bmnqw,nk->bmkqw',
                        tf.reduce_sum(tf.cast(tf.abs(h_freq), tf.float16), axis=[1,2]),
                        mask_UE
                    )
                )
            ),
            noise_power := (tx_power / (10.0 ** (snr_dB / 10.0))) +
            tx_power * tf.einsum(
                'xb,abcde->axcde',
                eye_BS_interference,
                tf.math.square(
                    tf.einsum(
                        'lm,bmkqw->blkqw',
                        mask_BS,
                        tf.einsum(
                            'bmnqw,nk->bmkqw',
                            tf.reduce_sum(tf.cast(tf.abs(h_freq), tf.float16), axis=[1,2]),
                            mask_UE
                        )
                    )
                )
            )
        )  # shape: [B, N_BS_Serving, N_UE_Active, N_sym, N_FFT]

        # EESM
        N_BS_Serving = sinr.shape[1]
        N_UE_Active = sinr.shape[2]
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(-sinr_flat / beta, axis=-1)
        sinr_eff = -beta * (log_sum_exp - tf.math.log(M))

        return h_delay, h_freq, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_TC8(self, topology, tau, mask_UE, mask_BS, eye_BS_interference, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):

        # TC2: FFT DFT 곱 사용 & BS,UE Masking  matmul 사용
        # TC3: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용
        # TC4: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, Reshape 사용하여 Batched Matmul 연산
        # TC5: FFT DFT 곱 사용 & BS,UE Masking + SINR 계산 matmul 사용, einsum 사용
        # TC6: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용
        # TC7: FFT 내장함수 사용, BS,UE Masking + SINR 계산 einsum 사용, 그래프 통합 시도 -실패(FFT, einsum 모두 custom call -> 그래프 분리)
        # TC8: FFT DFT 곱 사용, BS,UE Masking + SINR 계산 einsum 사용 

        # Doppler matrix 계산

        # h_field_array_power_ shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT, N_r, N_t, N_sym]
            # Reshaped: [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        # zoa_delay, aoa_delay shape
            # Original: [B, N_BS, N_UE, N_Rays, N_FFT]
            # Reshaped: [B, N_Rays, N_FFT, N_BS, N_UE]
        B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE = h_field_array_power_.shape

        velocities = topology.velocities #[B, N_UE, 3]
        v_bar = tf.expand_dims(velocities, axis=1) #[B, 1, N_UE, 3]
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 2) # [B, 1, 1, N_UE, 3]
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 3) # [B, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # [B, 1, 1, 1, N_UE, 3] or [B, 1, 1, N_UE, 1, 3]
        v_bar = tf.expand_dims(v_bar, 1) # 최종적으로 DL [B, 1, 1, 1, 1, N_UE, 3] or UL [B, 1, 1, 1, N_UE, 1, 3]
        r_hat_rx = self._unit_sphere_vector_Modified(zoa_delay, aoa_delay) # [B, 1, N_Rays, N_FFT, N_BS, N_UE, 3], axis=1: N_sym

        # [B, 1, N_Rays, N_FFT, N_BS, N_UE] * [1,N_sym,1,1,1,1] = [B, N_sym, N_Rays, N_FFT, N_BS, N_UE]
        sample_len = tf.shape(sample_times)[0]
        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -1) * tf.reshape(sample_times, [1,sample_len,1,1,1,1])
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -3), -3) # [B, N_sym, N_Rays, N_FFT, 1, 1, N_BS, N_UE] axis = -3,-4 for N_r, N_t

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_field_array_power = h_field_array_power_
        h_delay = h_field_array_power*h_doppler 

        # FFT 내장함수 사용X, DFT Matrix Element-wise Multiplication 
        # [N_FFT]
        frequencies = self.carrier_frequency * tf.range(-N_FFT/2, N_FFT/2, dtype=self.rdtype) 
        # [1,1,1,N_FFT,1,1,1,1]
        frequencies = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(frequencies, 0), 0), 0), -1), -1), -1), -1) 
        # [B, 1, N_Rays, 1, N_BS, N_UE] -> [B, 1, N_Rays, 1, 1, 1, N_BS, N_UE]
        tau = tf.expand_dims(tf.expand_dims(tau, -3), -3)

        # [B, 1, N_Rays, N_FFT, 1, 1, N_BS, N_UE]
        DFT_exp_factor = tf.exp(tf.complex(tf.constant(0, self.rdtype),-2*PI*frequencies*tau))

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = h_delay*DFT_exp_factor

        # [B, N_sym, N_Rays, N_FFT, N_r, N_t, N_BS, N_UE] -> # [B, N_sym, N_FFT, N_r, N_t, N_BS, N_UE]
        h_freq = tf.reduce_sum(h_freq, axis=2) # rays 합산 in frequency domain

        h_freq_abs = tf.reduce_sum(tf.cast(tf.abs(h_freq), tf.float16), axis=[3,4]) # [B, N_sym, N_FFT, N_BS, N_UE]
        h_freq_abs_active = tf.matmul(h_freq_abs, mask_UE) # [B, N_sym, N_FFT, N_BS, N_UE_Active]
        h_freq_abs_active_serving = tf.matmul(mask_BS, h_freq_abs_active) # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]

        h_freq_power = tf.math.square(h_freq_abs_active_serving)
        interference = tf.matmul(eye_BS_interference, h_freq_power)

        noise_power = tx_power / (10.0 ** (snr_dB / 10.0))

        # [B, N_sym, N_FFT, N_BS_Serving, N_UE_Active]
        sinr = (tx_power * h_freq_power) / (tx_power * interference + noise_power)


        # EESM
        N_BS_Serving = sinr.shape[-2]
        N_UE_Active = sinr.shape[-1]
        M = tf.cast(N_sym * N_FFT, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS_Serving, N_UE_Active, -1])
        log_sum_exp = tf.reduce_logsumexp(-sinr_flat / beta, axis=-1)
        sinr_eff = -beta * (log_sum_exp - tf.math.log(M))

        return h_delay, h_freq, sinr, sinr_eff

    @tf.function(jit_compile=True)
    def _H_TTI_TC_Final(self, topology, ActiveUE, ServingBS, sample_times, h_field_array_power_, aoa_delay, zoa_delay, snr_dB=10.0, tx_power=1.0, beta=1.0):
    
        # Doppler matrix 계산

        velocities = topology.velocities
        v_bar = tf.expand_dims(velocities, axis=-1)
        if topology.moving_end == 'rx':
            v_bar = tf.expand_dims(v_bar, 1)
        elif topology.moving_end == 'tx':
            v_bar = tf.expand_dims(v_bar, 2)
        v_bar = tf.expand_dims(v_bar, -3)
        v_bar = tf.expand_dims(v_bar, -3)
        
        r_hat_rx = self._unit_sphere_vector(zoa_delay, aoa_delay) #sin, cos 호출

        exponent = 2*PI/self._lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -2)*sample_times
        h_doppler = tf.exp(tf.complex(tf.constant(0., self.rdtype), exponent))
        h_doppler = tf.expand_dims(tf.expand_dims(h_doppler, -2), -2)


        # Power scaling & 최종 채널 계수 계산       

        h_field_array_power = h_field_array_power_

        # [B, N_BS, N_UE, N_rays, N_fft, N_rxAnt, N_txAnt, N_sym]
        h_field_array_power_doppler = h_field_array_power*h_doppler
        
        # [B, N_BS, N_UE, N_rays, N_fft, N_rxAnt, N_txAnt, N_sym] -> [B, N_BS, N_UE, N_fft, N_rxAnt, N_txAnt, N_sym]
        h_delay_active = tf.reduce_sum(h_field_array_power_doppler, axis=3)

        # [B, N_BS, N_UE, N_fft, N_rxAnt, N_txAnt, N_sym] -> [B, N_BS, N_UE, N_rxAnt, N_txAnt, N_sym, N_fft]
        h_delay_active = tf.transpose(h_delay_active, [0,1,2,4,5,6,3]) 

        # FFT

        h_freq_serving = tf.signal.fft(h_delay_active) #  [B, N_BS, N_UE, N_rxAnt, N_txAnt, N_sym, N_fft]
        
        # SINR

        # ======= 기본 입력 =======
        # h_freq_serving: [B, N_BS, N_UE, N_r, N_t, N_sym, N_fft]
        # snr_dB: scalar or [B, N_BS, N_UE, N_s, N_sym, N_fft]
        # tx_power: scalar or [B, N_BS, N_UE, N_s, N_sym, N_fft]
        # beta: scalar

        # ======= 0. 차원 정보 추출 =======
        B       = tf.shape(h_freq_serving)[0]
        N_BS    = tf.shape(h_freq_serving)[1]
        N_UE    = tf.shape(h_freq_serving)[2]
        N_r     = tf.shape(h_freq_serving)[3]
        N_t     = tf.shape(h_freq_serving)[4]
        N_sym   = tf.shape(h_freq_serving)[5]
        N_fft   = tf.shape(h_freq_serving)[6]
        N_s     = tf.minimum(N_t, N_r)  # 스트림 수 자동 결정 가능 (필요 시 고정값 사용)
        
        # ======= 1. Precoder/Detector 항등행렬로 초기화 =======
        # 사용자 정의 W_precoding 또는 G_detector가 없는 경우에만 기본 설정

        # [B, N_BS, N_UE, N_t, N_s, N_sym, N_fft]
        W_precoding = tf.eye(N_s, batch_shape=[B, N_BS, N_UE, 1, 1], dtype=tf.complex64)  # [B,N_BS,N_UE,N_s,N_s]
        W_precoding = tf.reshape(W_precoding, [B, N_BS, N_UE, N_t, N_s, 1, 1])
        W_precoding = tf.tile(W_precoding, [1, 1, 1, 1, 1, N_sym, N_fft])
        W_precoding = tf.transpose(W_precoding, [0, 1, 2, 4, 3, 5, 6])  # → [B, N_BS, N_UE, N_t, N_s, N_sym, N_fft]

        # [B, N_BS, N_UE, N_s, N_r, N_sym, N_fft]
        G_detector = tf.eye(N_s, batch_shape=[B, N_BS, N_UE, 1, 1], dtype=tf.complex64)
        G_detector = tf.reshape(G_detector, [B, N_BS, N_UE, N_s, N_r, 1, 1])
        G_detector = tf.tile(G_detector, [1, 1, 1, 1, 1, N_sym, N_fft])
        G_detector = tf.transpose(G_detector, [0, 1, 2, 3, 4, 5, 6])  # → [B, N_BS, N_UE, N_s, N_r, N_sym, N_fft]

        # ======= 2. Effective channel: H_eff = G @ H @ W =======
        HW = tf.einsum('aburtmf,abutsmf->abursmf', h_freq_serving, W_precoding)
        GHW = tf.einsum('abusrmf,aburqmf->abuspqmf', G_detector, HW)
        
        """

        H_eff = h_freq_serving
        # ======= 3. SINR 계산 =======
        diag_H_eff = tf.linalg.diag_part(H_eff)  # [B, N_BS, N_UE, N_s, N_sym, N_fft]
        signal_power = tf.math.square(tf.abs(diag_H_eff)) # 자기 스트림만의 유효 신호 전력
        signal_power = tf.cast(signal_power, tf.float16)
        H_eff_abs2 = tf.math.square(tf.abs(H_eff))  # [B, N_BS, N_UE, N_s, N_s, N_sym, N_fft]        
        H_eff_abs2 = tf.cast(H_eff_abs2, tf.float16)

        total_power = tf.reduce_sum(H_eff_abs2, axis=4) # 모든 스트림으로부터의 총 전력
        interference_power = total_power - signal_power # 간섭 전력

        g_norm2 = tf.reduce_sum(tf.math.square(tf.abs(G_detector)), axis=4)  # ||g_i||^2
        snr_lin = 10.0 ** (snr_dB / 10.0)
        noise_power = tx_power / snr_lin
        noise_term = noise_power * g_norm2

        sinr = (tx_power * signal_power) / (tx_power * interference_power + noise_term)
        # shape: [B, N_BS, N_UE, N_s, N_sym, N_fft]

        # ======= 4. EESM 계산 =======
        M = tf.cast(N_sym * N_fft, sinr.dtype)
        sinr_flat = tf.reshape(sinr, [B, N_BS, N_UE, N_s, -1])
        log_sum_exp = tf.reduce_logsumexp(- sinr_flat / beta, axis=-1)
        sinr_eff = - beta * (log_sum_exp - tf.math.log(M))
        # 최종 shape: [B, N_BS, N_UE, N_s]        
        """ 


        return  h_delay_active, h_freq_serving, GHW
    
# End of this code